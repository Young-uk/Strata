"""Model-free rollback and exclusive compute ownership checks."""
import os
import threading
import unittest
from pathlib import Path
from serve.atomic_summary import SummaryLane, SummaryError
from serve.priority import WeightedScheduler
from serve.test_priority_scheduler import exhaust

class Engine:
    def __init__(self):
        self.main = 'original'
        self.calls = []
        self.fail_restore = False
    def session_file(self, action, path):
        self.calls.append(action)
        if action == 'save':
            Path(path).write_text(self.main)
        else:
            if self.fail_restore:
                raise RuntimeError('restore uncertain')
            self.main = Path(path).read_text()
        return {'tokens': 3, 'bytes': 8}

class Recovery(unittest.TestCase):
    def setUp(self):
        self.lane = SummaryLane()
        self.scheduler = WeightedScheduler(threading.Lock())
        self.lane.scheduler = self.scheduler
        self.cancel = threading.Event()
        self.owner = exhaust(self.scheduler.admit(True, self.cancel, recovery=True))
        self.cancel._priority_owner = self.owner
        self.cancel._atomic_clone_snapshot = {'ids': [1,2,3], 'tokens': 3, 'tail': 4}
        self.engine = Engine()
    def test_clean_bank_memory_refusal_uses_disk_only_for_explicit_urgent_recovery(self):
        self.lane.mode = 'state-bank-v1'
        self.lane.max_slot_bytes = 100
        self.lane.disk_recovery_enabled = True
        def refused(action, *args, **kwargs):
            self.engine.calls.append(action)
            error = RuntimeError('normal pressure, stable swap and RAM floor required')
            error.summary_refused = True
            raise error
        self.engine.summary_state = refused
        with self.lane.transaction(self.engine, self.cancel):
            self.engine.main = 'summary'
        self.assertEqual(self.engine.main, 'original')
        self.assertEqual(self.engine.calls, ['fork','save','restore'])
        self.assertIsNone(self.scheduler.exclusive_owner)
        self.assertFalse(self.lane.poisoned)

    def test_cancellation_or_generation_error_restores_before_unsealing(self):
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with self.lane.serial_recovery(self.engine, self.cancel):
                self.engine.main = 'summary'
                self.cancel.set()
                raise RuntimeError('generation failed')
        self.assertEqual(self.engine.main, 'original')
        self.assertEqual(self.engine.calls, ['save','restore'])
        self.assertIsNone(self.scheduler.exclusive_owner)
        self.assertFalse(self.lane.poisoned)
    def test_foreground_never_runs_between_serial_turns(self):
        with self.lane.serial_recovery(self.engine, self.cancel):
            fg = exhaust(self.scheduler.admit(False, threading.Event()))
            fg.ready = True
            self.assertIsNone(self.scheduler._next())
            self.owner.ready = True
            self.assertIs(self.scheduler._next(), self.owner)
            self.owner.ready = False
        self.assertIs(self.scheduler._next(), fg)
    def test_restore_failure_keeps_disk_artifact_and_blocks_inference(self):
        self.engine.fail_restore = True
        with self.assertRaises(SummaryError):
            with self.lane.serial_recovery(self.engine, self.cancel):
                self.engine.main = 'summary'
        self.assertTrue(self.lane.poisoned)
        self.assertIsNotNone(self.scheduler.poison_reason)
        self.assertEqual(Path(self.lane.recovery_path).read_text(), 'original')
        os.unlink(self.lane.recovery_path)
    def test_no_receipt_never_changes_main(self):
        self.cancel._atomic_clone_snapshot.pop('ids')
        with self.assertRaises(SummaryError):
            with self.lane.serial_recovery(self.engine, self.cancel):
                self.fail('must not generate')
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.engine.main, 'original')
