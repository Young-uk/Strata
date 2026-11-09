"""Fake native protocol checks. No subprocess, GPU, model or server listener."""
import contextlib
import queue
import threading
import unittest
from unittest.mock import patch
from serve.atomic_summary import SummaryLane
from serve.priority import WeightedScheduler, PriorityError
from serve.priority_client import ContinuationClient
from serve.test_priority_scheduler import exhaust

class Engine:
    gen = 1
    silence_s = 10
    def __init__(self):
        self.lines = queue.Queue(); self.sent = []; self.steps = 0; self.bad = None
    def alive(self): return True
    def _exclusive_control(self): return contextlib.nullcontext()
    def sampling_keys(self, sampling): return ''
    def _parse_done(self, line):
        f = line.split()
        self.last = {'generated':int(f[1]),'prompt_tokens':int(f[2]),'prompt_ms':float(f[3]),
            'decode_ms':float(f[4]),'reused':int(f[8]) if len(f)>8 else 0,
            'prompt_read':int(f[14]) if len(f)>14 else int(f[2]),'finish':f[5]}
    def _send_for(self, generation, command):
        self.sent.append(command)
        pieces = command.split(); kind = pieces[0]
        identity = pieces[1] if len(pieces) > 1 else ''
        if kind == 'AGEN': self.lines.put(f'AREADY {identity} 0 3')
        if kind == 'APREP':
            self.steps = 1
            if self.bad:
                self.lines.put(self.bad.replace('{id}',identity)); return
            self.lines.put('T 101')
            self.lines.put('DONE 1 3 10 2 length')
            self.lines.put(f'APREPARED {identity} 3 1 active')
        if kind == 'ANEXT':
            self.steps += 1
            if self.bad:
                self.lines.put(self.bad.replace('{id}', identity)); return
            self.lines.put(f'AT {identity} {100 + self.steps}')
            self.lines.put(f'AYIELD {identity} decode 3 1' if self.steps == 1 else f'ADONE {identity} length 2')
        if kind == 'ACANCEL': self.lines.put(f'ACANCELLED {identity}')
        if kind == 'ASYNC_CAPS': self.lines.put('ASYNC_CAPS 1 512 1 text')
        if kind == 'APREP_CAPS': self.lines.put('APREP_CAPS 1 legacy1')

class ClientTests(unittest.TestCase):
    def setUp(self):
        self.e = Engine(); self.lane = SummaryLane(); self.s = WeightedScheduler(threading.Lock())
        self.cancel = threading.Event(); self.owner = exhaust(self.s.admit(False, self.cancel))
        self.cancel._priority_owner = self.owner
        self.client = ContinuationClient(self.e, self.s, self.lane)

    def test_serial_recovery_appends_to_main_without_forking_a_bank(self):
        self.s.close(self.owner)
        self.owner = exhaust(self.s.admit(True, self.cancel, recovery=True))
        self.cancel._priority_owner = self.owner
        self.cancel._atomic_serial_recovery = {"ids": [1, 2], "tokens": 2, "tail": 3}
        original = self.e._send_for
        def send(generation, command):
            identity = command.split()[1]
            if command.startswith('ARECOVER '):
                self.e.sent.append(command)
                self.e.lines.put(f'AREADY {identity} 2 4')
            elif command.startswith('ANEXT '):
                self.e.sent.append(command)
                self.e.lines.put(f'AT {identity} 101')
                self.e.lines.put(f'ADONE {identity} length 1')
            else:
                original(generation, command)
        self.e._send_for = send
        self.assertEqual(list(self.client.generate([4], 1, {}, self.cancel)), [101])
        self.assertTrue(self.e.sent[0].startswith('ARECOVER '))
        self.assertIn(' main 1 ', self.e.sent[0])
        self.assertTrue(self.e.sent[0].endswith('1,2,3,4'))
        self.assertFalse(any(command.startswith('AFORK ') for command in self.e.sent))

    def test_capability_exact(self): self.client.discover()

    def test_two_tokens_no_stop_or_replay(self):
        output = list(self.client.generate([1,2,3], 2, {}, self.cancel))
        self.assertEqual(output, [101,102])
        self.assertEqual([s.split()[0] for s in self.e.sent], ['APREP','ANEXT'])
        self.assertEqual(self.e.last['generated'], 2)
        self.assertEqual(self.e.last['prompt_ms'], 10.0)
        self.assertIsNone(self.e.last['decode_ms'])
        self.assertIsNone(self.s.turn_owner)

    def test_disconnect_after_token_cancels_only_owned_continuation(self):
        gen = self.client.generate([1,2,3], 2, {}, self.cancel)
        self.assertEqual(next(gen), 101); gen.close()
        self.assertEqual([s.split()[0] for s in self.e.sent], ['APREP','ACANCEL'])
        self.assertIsNone(self.s.turn_owner)
        self.assertFalse(self.lane.poisoned)

    def test_foreground_cancel_does_not_cancel_summary_owner(self):
        bg_cancel = threading.Event(); bg = exhaust(self.s.admit(True, bg_cancel))
        gen = self.client.generate([1,2,3], 2, {}, self.cancel)
        self.assertEqual(next(gen), 101); self.cancel.set(); list(gen)
        self.assertFalse(bg.cancel.is_set())
        self.assertEqual(self.e.sent[-1].split()[0], 'ACANCEL')

    def test_wrong_id_poisoned_without_followup_command(self):
        self.e.bad = 'APREPARED wrong 3 1 active'
        with self.assertRaises(PriorityError): list(self.client.generate([1,2,3], 2, {}, self.cancel))
        self.assertTrue(self.lane.poisoned)
        self.assertNotIn('ACANCEL', [s.split()[0] for s in self.e.sent])

    def test_nonmonotonic_progress_poisoned(self):
        self.e.bad = 'APREPARED {id} 3 7 active'
        with self.assertRaises(PriorityError): list(self.client.generate([1,2,3], 2, {}, self.cancel))
        self.assertTrue(self.lane.poisoned)

    def test_generation_replaced_fails_before_send(self):
        self.e.gen = 2
        with self.assertRaises(PriorityError): list(self.client.generate([1,2,3], 2, {}, self.cancel))
        self.assertFalse(self.e.sent)

    def test_disconnect_during_quantum_heartbeat_drains_then_cancels(self):
        # Inject a heartbeat before the queued AT/AYIELD, simulating a quiet GPU.
        self.s.close(self.owner)
        self.owner = exhaust(self.s.admit(True, self.cancel))
        self.cancel._priority_owner = self.owner
        original = self.client._read
        def heartbeat(identity=None):
            if self.e.sent[-1].startswith('ANEXT'):
                yield None
            return (yield from original(identity))
        with patch.object(self.client, '_read', heartbeat):
            gen = self.client.generate([1,2,3], 2, {}, self.cancel)
            self.assertIsNone(next(gen)); self.assertIs(self.s.turn_owner, self.owner)
            gen.close()
        self.assertIsNone(self.s.turn_owner)
        self.assertEqual(self.e.sent[-1].split()[0], 'ACANCEL')
        self.assertTrue(self.e.lines.empty())



class PreparationTests(ClientTests):
    def test_native_reused_marker_before_first_token(self):
        # The real legacy GEN body always emits RESUME before prefill and
        # REUSED after it. APREP retains both before its tagged boundary.
        original = self.e._send_for
        def send(generation, command):
            if command.startswith('APREP '):
                self.e.lines.put('RESUME 2')
                self.e.lines.put('PP 3 3 10')
                self.e.lines.put('REUSED 2')
            original(generation, command)
        self.e._send_for = send
        self.assertEqual([t for t in self.client.generate([1,2,3],2,{},self.cancel) if t is not None], [101,102])
        self.assertFalse(self.lane.poisoned)
        self.assertTrue(self.e.lines.empty())
        self.assertEqual(self.e.reused, 2)
        self.assertEqual([s.split()[0] for s in self.e.sent], ['APREP','ANEXT'])

    def test_native_reused_marker_during_cancel_drain(self):
        original = self.e._send_for
        def send(generation, command):
            if command.startswith('APREP '):
                self.e.sent.append(command)
                identity = command.split()[1]
                for line in ('RESUME 0', 'PP 1 3 10', 'REUSED 0',
                             'DONE 0 3 10 0 cancel 0 0 0 0 0 0 0 0 1',
                             f'APREPARED {identity} 1 0 cancel'):
                    self.e.lines.put(line)
            else:
                original(generation, command)
        self.e._send_for = send
        gen = self.client.generate([1,2,3],2,{},self.cancel)
        self.assertIsNone(next(gen))
        gen.close()
        self.assertFalse(self.lane.poisoned)
        self.assertTrue(self.e.lines.empty())
        self.assertIsNone(self.s.turn_owner)
        self.assertEqual(self.e.last['prompt_read'], 1)
        self.assertEqual(self.e.last['finish'], 'cancel')
        self.assertEqual([s.split()[0] for s in self.e.sent], ['APREP','STOP'])

    def test_invalid_reused_count_remains_fail_closed(self):
        for marker in ('REUSED -1', 'REUSED 4', 'REUSED bad', 'REUSED 0 extra'):
            with self.subTest(marker=marker):
                self.setUp()
                self.e.bad = marker
                with self.assertRaises(PriorityError):
                    list(self.client.generate([1,2,3],2,{},self.cancel))
                self.assertTrue(self.lane.poisoned)

    def test_preparation_cancel_keeps_partial_prompt_count(self):
        original = self.e._send_for
        def send(generation, command):
            if command.startswith('APREP'):
                self.e.sent.append(command); identity=command.split()[1]
                self.e.lines.put('PP 1 3 10')
                self.e.lines.put('DONE 0 3 10 0 cancel 0 0 0 0 0 0 0 0 1')
                self.e.lines.put(f'APREPARED {identity} 1 0 cancel')
            else: original(generation, command)
        self.e._send_for=send
        gen=self.client.generate([1,2,3],2,{},self.cancel)
        self.assertIsNone(next(gen)); gen.close()
        self.assertEqual(self.e.last['prompt_read'],1)
        self.assertEqual(self.e.last['finish'],'cancel')
        self.assertEqual([s.split()[0] for s in self.e.sent],['APREP','STOP'])
        self.assertTrue(self.e.lines.empty())

    def test_cancel_after_prepared_token_not_length(self):
        gen=self.client.generate([1,2,3],2,{},self.cancel)
        self.assertEqual(next(gen),101); gen.close()
        self.assertEqual(self.e.last['finish'],'cancel')
        self.assertEqual(self.e.last['generated'],1)

    def test_image_preparation_uses_original_geni_path(self):
        self.assertEqual(list(self.client.generate([1,2,3],2,{},self.cancel,embeddings='/tmp/image.bin')),[101,102])
        self.assertIn(' GENI /tmp/image.bin ',self.e.sent[0])

    def test_completed_image_request_binds_processed_state_for_text_clone(self):
        original = self.e._send_for
        def send(generation, command):
            if command.startswith('ANEXT'):
                identity = command.split()[1]
                self.e.sent.append(command)
                self.e.lines.put(f'AT {identity} 102')
                self.e.lines.put(f'ADONE {identity} stop 2')
            else:
                original(generation, command)
        self.e._send_for = send
        output = list(self.client.generate([1, 2, 3], 2, {'atomic_cache_owner': 'thread/image'},
                                          self.cancel, embeddings='/tmp/image.bin'))
        self.assertEqual(output, [101, 102])
        snapshot = self.e.atomic_main_snapshot
        self.assertEqual(snapshot['owner'], 'thread/image')
        self.assertEqual(snapshot['ids'], [1, 2, 3, 101])
        self.assertEqual(snapshot['tail'], 102)
        self.assertNotIn('embeddings', snapshot)

    def test_failed_preparation_is_clean_error_not_false_complete(self):
        original=self.e._send_for
        def send(generation,command):
            if command.startswith('APREP'):
                self.e.sent.append(command);identity=command.split()[1]
                self.e.lines.put('ERR unsupported option')
                self.e.lines.put(f'APREPARED {identity} 0 0 failed')
            else:original(generation,command)
        self.e._send_for=send
        with self.assertRaisesRegex(ValueError,'unsupported option'):
            list(self.client.generate([1,2,3],2,{},self.cancel))
        self.assertFalse(self.lane.poisoned)
        self.assertEqual(self.e.last['finish'],'error')
        self.assertEqual(self.e.last['prompt_read'],0)


class ConcurrentProtocolTests(unittest.TestCase):
    def test_summary_advances_before_foreground_finishes_without_stop(self):
        import time
        class MultiEngine(Engine):
            def __init__(self):
                super().__init__(); self.records={}; self.events=[]; self.guard=threading.Lock()
            def _send_for(self,generation,command):
                self.assertion = self.guard.acquire(blocking=False)
                if not self.assertion: raise AssertionError('overlapping native command')
                try:
                    time.sleep(.001)
                    self.sent.append(command); fields=command.split();kind=fields[0];identity=fields[1]
                    if kind=='APREP':
                        self.records[identity]=['main',1,int(fields[2])]
                        self.lines.put('T 101');self.lines.put('DONE 1 3 10 2 length')
                        self.lines.put(f'APREPARED {identity} 3 1 active')
                    elif kind=='AGEN':
                        self.records[identity]=['summary',0,int(fields[3])]
                        self.lines.put(f'AREADY {identity} 0 3')
                    elif kind=='ANEXT':
                        row=self.records[identity];row[1]+=1;count=row[1]
                        self.events.append((row[0],count))
                        self.lines.put(f'AT {identity} {100+count}')
                        self.lines.put(f'ADONE {identity} length {count}' if count==row[2] else
                                       f'AYIELD {identity} decode {3+count-1} {count}')
                    elif kind=='ACANCEL':self.lines.put(f'ACANCELLED {identity}')
                finally:self.guard.release()
        engine=MultiEngine();lane=SummaryLane();scheduler=WeightedScheduler(threading.Lock())
        client=ContinuationClient(engine,scheduler,lane)
        outcomes=[];errors=[];start=threading.Barrier(2)
        def run(summary,limit):
            cancel=threading.Event();owner=exhaust(scheduler.admit(summary,cancel));cancel._priority_owner=owner
            try:
                start.wait()
                outcomes.append((summary,[token for token in client.generate([1,2,3],limit,{},cancel) if token is not None]))
            except BaseException as exc:errors.append(exc)
            finally:scheduler.close(owner)
        a=threading.Thread(target=run,args=(False,64));b=threading.Thread(target=run,args=(True,8))
        a.start();b.start();a.join(5);b.join(5)
        self.assertFalse(a.is_alive() or b.is_alive());self.assertFalse(errors)
        first_summary=next(i for i,row in enumerate(engine.events) if row[0]=='summary')
        main_done=engine.events.index(('main',64))
        self.assertLess(first_summary,main_done)
        self.assertFalse(any(cmd.startswith('STOP') for cmd in engine.sent))
        self.assertFalse(lane.poisoned)
        self.assertEqual(sorted((summary,len(tokens)) for summary,tokens in outcomes),[(False,64),(True,8)])

if __name__ == '__main__': unittest.main()
