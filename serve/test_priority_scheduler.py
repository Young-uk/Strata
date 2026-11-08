"""CPU-only scheduling/ownership checks; never imports or runs a model."""
import threading
import unittest
from serve.priority import WeightedScheduler, PriorityCancelled, PriorityError, RequestValue, observation


def exhaust(generator):
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = WeightedScheduler(threading.Lock(), 8)
        self.fg = exhaust(self.scheduler.admit(False, threading.Event()))
        self.bg = exhaust(self.scheduler.admit(True, threading.Event()))

    def test_summary_positive_share_with_foreground_ready(self):
        selected = []
        for _ in range(27):
            self.fg.ready = self.bg.ready = True
            owner = self.scheduler._next()
            release = exhaust(self.scheduler.turn(owner))
            selected.append('S' if owner.summary else 'F')
            release()
        self.assertEqual(''.join(selected), ('FFFFFFFFS' * 3))
        self.assertEqual(self.bg.turns, 3)

    def test_only_ready_owner_uses_all_turns(self):
        for _ in range(12):
            release = exhaust(self.scheduler.turn(self.bg)); release()
        self.assertEqual(self.bg.turns, 12)
        self.assertEqual(self.fg.turns, 0)

    def test_foreground_arrival_never_cancels_summary(self):
        release = exhaust(self.scheduler.turn(self.bg))
        self.assertFalse(self.bg.cancel.is_set())
        self.fg.ready = True
        self.assertIs(self.scheduler.turn_owner, self.bg)
        release()
        self.assertIs(self.scheduler._next(), self.fg)
        self.assertFalse(self.bg.cancel.is_set())

    def test_cancelled_request_never_takes_quantum(self):
        self.bg.cancel.set()
        with self.assertRaises(PriorityCancelled):
            exhaust(self.scheduler.turn(self.bg))
        self.assertIsNone(self.scheduler.turn_owner)
        self.assertFalse(self.bg.ready)
        # Its acknowledged native continuation still needs owner-scoped cleanup.
        release = exhaust(self.scheduler.turn(self.bg, cleanup=True)); release()

    def test_double_release_and_stale_owner(self):
        release = exhaust(self.scheduler.turn(self.fg)); release(); release()
        self.scheduler.close(self.fg)
        with self.assertRaises(PriorityError):
            exhaust(self.scheduler.turn(self.fg))

    def test_closing_inflight_owner_poisoned(self):
        release = exhaust(self.scheduler.turn(self.fg))
        with self.assertRaises(PriorityError):
            self.scheduler.close(self.fg)
        self.assertIsNotNone(self.scheduler.poison_reason)
        release()

    def test_lane_waiter_disconnect_does_not_steal_owner(self):
        cancel = threading.Event()
        request = self.scheduler.admit(False, cancel)
        next(request)
        cancel.set()
        with self.assertRaises(PriorityCancelled):
            next(request)
        self.assertIs(self.scheduler.owners[False], self.fg)
        self.assertFalse(self.scheduler.waiters[False])

    def test_closing_waiter_cleans_registration(self):
        request = self.scheduler.admit(False, threading.Event())
        next(request); request.close()
        self.assertFalse(self.scheduler.waiters[False])

    def test_observation_isolation_and_legacy_values(self):
        class Engine:
            progress = RequestValue()
        engine = Engine(); engine.progress = ('old', 1)
        barrier = threading.Barrier(2); result = []
        def run(value):
            with observation(engine):
                engine.progress = value
                barrier.wait()
                result.append(engine.progress)
        a = threading.Thread(target=run, args=((10, 20),))
        b = threading.Thread(target=run, args=((30, 40),))
        a.start(); b.start(); a.join(); b.join()
        self.assertCountEqual(result, [(10, 20), (30, 40)])
        self.assertIn(engine.progress, result)


class BankTransitionTests(unittest.TestCase):
    def make(self):
        from serve.atomic_summary import SummaryLane
        from serve.test_summary_state_bank import Bank
        engine = Bank(); lane = SummaryLane(); lane.configure_state_bank(engine, 1024)
        lane.scheduler = WeightedScheduler(threading.Lock())
        cancel = threading.Event()
        owner = exhaust(lane.scheduler.admit(True, cancel)); cancel._priority_owner = owner
        return engine, lane, cancel, owner

    def test_foreground_progress_changes_token_count_not_identity(self):
        engine, lane, cancel, owner = self.make()
        original = engine.summary_state
        def state(action, *args, **kwargs):
            result = original(action, *args, **kwargs)
            if action == 'end': result['main_tokens'] += 20
            return result
        engine.summary_state = state
        with lane.transaction(engine, cancel):
            lane.foreground_arrived()
            self.assertFalse(cancel.is_set())
        self.assertFalse(lane.poisoned)

    def test_cancel_before_begin_permit_does_not_poison(self):
        from serve.atomic_summary import SummaryError
        engine, lane, cancel, owner = self.make()
        lane.scheduler.fifo.acquire()
        result = []
        def run():
            try:
                with lane.transaction(engine, cancel): pass
            except SummaryError as exc: result.append(exc.code)
        worker = threading.Thread(target=run); worker.start()
        for _ in range(100):
            if lane.status()['active']: break
            threading.Event().wait(.005)
        cancel.set(); worker.join(1)
        lane.scheduler.fifo.release()
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, ['atomic_summary_interrupted'])
        self.assertFalse(lane.poisoned)
        self.assertIsNone(lane.scheduler.poison_reason)
        self.assertEqual(engine.calls, ['caps'])

    def test_uncertain_begin_blocks_existing_other_owner(self):
        from serve.atomic_summary import SummaryError
        engine, lane, cancel, owner = self.make()
        foreground = exhaust(lane.scheduler.admit(False, threading.Event()))
        engine.fail = 'begin'
        with self.assertRaises(SummaryError):
            with lane.transaction(engine, cancel): pass
        with self.assertRaises(PriorityError): exhaust(lane.scheduler.turn(foreground))
        self.assertTrue(lane.poisoned)

    def test_uncertain_end_blocks_existing_other_owner(self):
        from serve.atomic_summary import SummaryError
        engine, lane, cancel, owner = self.make()
        foreground = exhaust(lane.scheduler.admit(False, threading.Event()))
        with self.assertRaises(SummaryError):
            with lane.transaction(engine, cancel): engine.fail = 'mismatch'
        with self.assertRaises(PriorityError): exhaust(lane.scheduler.turn(foreground))

class ServiceControlTests(unittest.TestCase):
    def setUp(self):
        from serve.server import Service, ByteTokenizer, MockEngine
        from serve.frontend import ChatTemplate
        from pathlib import Path
        tok = ByteTokenizer(); engine = MockEngine(tok, 'answer')
        self.svc = Service(engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        self.svc.summary_lane.scheduler = WeightedScheduler(self.svc.fifo)

    def test_global_controls_explicitly_unsupported(self):
        from unittest.mock import patch
        svc = self.svc
        svc.slot_save_path = '/tmp'
        with patch.object(svc.engine, 'session_file', create=True) as session:
            self.assertEqual(svc.slot_action('0', 'save', 'example')[0], 501)
            session.assert_not_called()
        with self.assertRaises(ValueError): svc.vram(None)
        self.assertEqual(svc.unload(), 'unsupported')
        with patch.object(svc.engine, 'end_for_restart', create=True) as end:
            self.assertFalse(svc.end_engine_if_unchanged(1)); end.assert_not_called()

    def test_cancelled_lane_wait_visible_and_clean(self):
        from serve.test_priority_scheduler import exhaust
        svc = self.svc; scheduler = svc.summary_lane.scheduler
        first = exhaust(scheduler.admit(False, threading.Event()))
        cancel = threading.Event(); run = svc.run([1], False, [], 1, {}, cancel)
        kind, _ = next(run); self.assertEqual(kind, 'ping')
        self.assertEqual(svc.status['queued'], 1)
        self.assertEqual(len(svc.terminal_requests), 1)
        cancel.set(); events = list(run)
        self.assertEqual(events[-1][1]['finish'], 'cancel')
        self.assertEqual(svc.status['queued'], 0)
        self.assertFalse(svc.terminal_requests)
        self.assertIs(scheduler.owners[False], first)


if __name__ == '__main__':
    unittest.main()
