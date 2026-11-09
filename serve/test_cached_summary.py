"""Bound clone admission and scheduling, using fake engines only."""
import threading
import unittest
from serve.atomic_summary import SummaryLane, request
from serve.test_summary_state_bank import ProtocolTests, Bank
from serve.priority import WeightedScheduler

class CachedSummaryTests(unittest.TestCase):
    def test_positive_clone_protocol_discovery(self):
        engine = ProtocolTests().engine(['SUMMARY_CLONE_CAPS 1 65536\n'])
        self.assertTrue(engine.summary_state('clone_caps'))
        self.assertEqual(engine.proc.sent, ['SUMMARY_CLONE_CAPS\n'])
        self.assertTrue(engine.summary_clone_supported)

    def test_bound_clone_wire_and_no_disk_fallback(self):
        engine = ProtocolTests().engine(['SUMMARY_STATE_HELD clone1 53000 1055000000 1\n'])
        binding = {'tokens': 53000, 'hash': 18446744073709551615}
        result = engine.summary_state('fork', 'clone1', max_slot_bytes=1300000000, binding=binding)
        self.assertEqual(result['main_tokens'], 53000)
        self.assertEqual(engine.proc.sent, ['SUMMARY_STATE_FORK 1300000000 clone1 53000 18446744073709551615\n'])
        for binding in (None, {}, {'tokens': True, 'hash': 1}, {'tokens': 1, 'hash': -1}):
            with self.assertRaises(ValueError): engine.summary_state('fork', 'x', max_slot_bytes=1, binding=binding)
        self.assertEqual(len(engine.proc.sent), 1)

    def test_wrong_conversation_refusal_is_clean(self):
        engine = ProtocolTests().engine(['SUMMARY_STATE_REFUSED same foreground cache changed before clone admission\n'])
        with self.assertRaises(Exception) as caught:
            engine.summary_state('fork', 'same', max_slot_bytes=1300000000, binding={'tokens': 50000, 'hash': 9})
        self.assertTrue(caught.exception.summary_refused)
        self.assertEqual(len(engine.proc.sent), 1)

    def test_lane_forks_with_exact_owner_binding_and_releases(self):
        engine = Bank(); calls = []; original = engine.summary_state
        def state(action, txid=None, **kw):
            calls.append((action, kw))
            return original('begin' if action == 'fork' else action, txid, **kw)
        engine.summary_state = state
        lane = SummaryLane(); lane.configure_state_bank(engine, 1300000000)
        cancel = threading.Event(); cancel._atomic_clone_snapshot = {'tokens': 53000, 'hash': 19, 'tail': 2, 'owner': 'thread/turn'}
        with lane.transaction(engine, cancel): pass
        self.assertEqual([action for action, _ in calls], ['caps', 'fork', 'end'])
        self.assertEqual(calls[1][1]['binding'], cancel._atomic_clone_snapshot)
        self.assertFalse(lane.poisoned)

    def test_idle_clone_uses_physical_chunks_and_yields_to_foreground(self):
        scheduler = WeightedScheduler(threading.Semaphore(1))
        def admit(summary):
            iterator = scheduler.admit(summary, threading.Event())
            while True:
                try: next(iterator)
                except StopIteration as result: return result.value
        owner = admit(True)
        turn = scheduler.turn(owner)
        while True:
            try: next(turn)
            except StopIteration as result: release = result.value; break
        self.assertEqual(scheduler.summary_input_quantum(owner), 128)
        self.assertEqual(scheduler.summary_input_quantum(owner, cached=True), 2048)
        foreground = admit(False)
        self.assertEqual(scheduler.summary_input_quantum(owner, cached=True), 1)
        release(); scheduler.close(foreground); scheduler.close(owner)

    def test_clone_owner_cannot_inject_native_commands(self):
        base = dict(atomic_summary=True, think=False, messages=[{'role':'user','content':'summarize'}])
        for owner in ('x\nQUIT', '', True, 'a'*257):
            with self.assertRaises(ValueError): request({**base, 'atomic_summary_clone': owner})
        self.assertEqual(request({**base, 'atomic_summary_clone': 'thread/user-1'})['atomic_summary_clone'], 'thread/user-1')

    def test_append_only_clone_continuation_never_sends_history(self):
        import contextlib
        import queue
        from serve.priority_client import ContinuationClient
        from serve.test_priority_scheduler import exhaust
        class Engine:
            gen=1; silence_s=10
            def __init__(self): self.lines=queue.Queue(); self.sent=[]; self.steps=0
            def sampling_keys(self, sampling): return ""
            def alive(self): return True
            def _exclusive_control(self): return contextlib.nullcontext()
            def _send_for(self, generation, command):
                self.sent.append(command); words=command.split(); identity=words[1]
                if words[0]=='AFORK': self.lines.put(f'AREADY {identity} 53000 53004')
                elif words[0]=='ANEXT':
                    self.steps+=1
                    if self.steps==1: self.lines.put(f'AYIELD {identity} prefill 53003 0')
                    else:
                        self.lines.put(f'AT {identity} 99'); self.lines.put(f'ADONE {identity} stop 1')
        engine=Engine(); scheduler=WeightedScheduler(threading.Semaphore(1)); lane=SummaryLane()
        cancel=threading.Event(); owner=exhaust(scheduler.admit(True,cancel)); cancel._priority_owner=owner
        cancel._atomic_clone_snapshot={'tokens':53000,'tail':2,'hash':42,'owner':'thread/turn'}
        client=ContinuationClient(engine,scheduler,lane)
        self.assertEqual(list(client.generate([10,11,12],1,{},cancel)),[None,99])
        self.assertEqual(engine.sent[0].split()[-1],'2,10,11,12')
        self.assertTrue(engine.sent[0].startswith('AFORK '))
        self.assertEqual(engine.last['prompt_tokens'],4)
        self.assertEqual(engine.last['prompt_read'],4)
        self.assertEqual(engine.last['cloned_tokens'],53000)
        self.assertEqual(cancel._atomic_input_progress[:3],(53003,53004,53000))
        self.assertIsNone(scheduler.poison_reason)

    def test_finished_foreground_snapshot_excludes_uncommitted_terminal_token(self):
        from serve.test_priority_client import Engine
        from serve.priority_client import ContinuationClient
        from serve.test_priority_scheduler import exhaust
        class FinishedEngine(Engine):
            def _send_for(self, generation, command):
                if command.startswith('ANEXT '):
                    self.sent.append(command); identity=command.split()[1]
                    self.lines.put(f'AT {identity} 2')
                    self.lines.put(f'ADONE {identity} stop 2')
                else: super()._send_for(generation, command)
        engine=FinishedEngine(); scheduler=WeightedScheduler(threading.Semaphore(1)); lane=SummaryLane()
        cancel=threading.Event(); owner=exhaust(scheduler.admit(False,cancel)); cancel._priority_owner=owner
        client=ContinuationClient(engine,scheduler,lane)
        self.assertEqual(list(client.generate([1,2,3],2,{'atomic_cache_owner':'thread/turn'},cancel)),[101,2])
        snapshot=engine.atomic_main_snapshot
        self.assertEqual(snapshot['owner'],'thread/turn')
        self.assertEqual(snapshot['tokens'],4)
        self.assertEqual(snapshot['tail'],2)
        self.assertEqual(snapshot['hash'],16341717103494134512)
