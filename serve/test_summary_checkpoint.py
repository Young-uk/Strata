"""Checkpoint recovery keeps archived coverage and protects the full main owner."""
import unittest
from serve.history_plan import summary_checkpoint, token_hash
from serve.test_summary_state_bank import ProtocolTests

class SummaryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(12000))
        self.main = dict(owner='room/turn', tokens=len(self.ids), ids=self.ids,
                         hash=token_hash(self.ids), tail=55, history_hash='a'*64)
        self.candidates = [(n, token_hash(self.ids[:n])) for n in (1000, 5000, 9000)]

    def plan(self, **kw):
        args = dict(snapshot=self.main, owner='room/turn', prefix=self.ids[:4000],
                    candidates=self.candidates, capacity=14000, task_tokens=2000)
        args.update(kw)
        return summary_checkpoint(**args)

    def test_earlier_complete_checkpoint_leaves_all_physical_output_room(self):
        plan = self.plan()
        self.assertEqual(plan['tokens'], 5000)
        self.assertEqual(plan['source_tokens'], 4000)
        self.assertEqual(plan['tail'], self.ids[5000])
        self.assertIs(plan['main_snapshot'], self.main)
        self.assertEqual(len(self.main['ids']), 12000)
        self.assertEqual(self.main['hash'], token_hash(self.ids))

    def test_never_omit_archived_tokens_or_accept_stale_binding(self):
        self.assertIsNone(self.plan(prefix=self.ids[:10000]))
        self.assertIsNone(self.plan(owner='other/turn'))
        self.assertIsNone(self.plan(prefix=[-1, *self.ids[1:4000]]))
        self.assertIsNone(self.plan(candidates=[(5000, 99)]))
        self.assertIsNone(self.plan(capacity=8000))
        self.assertIsNone(self.plan(candidates=[]))

    def test_image_runs_require_attested_completed_history(self):
        ids = [1, 2, 99, 99, 99, 3, 4, 5, 6]
        main = dict(owner='room/turn', tokens=9, ids=ids, hash=token_hash(ids), history_hash='a'*64)
        kw = dict(snapshot=main, prefix=[1, 2, 99, 3], candidates=[(7, token_hash(ids[:7]))],
                  image_pad=99, capacity=10000, task_tokens=1000)
        self.assertIsNone(self.plan(**kw))
        self.assertIsNone(self.plan(**kw, history_hash='b'*64))
        plan = self.plan(**kw, history_hash='a'*64)
        self.assertEqual(plan['source_tokens'], 6)
        self.assertEqual(plan['tokens'], 7)
        self.assertNotIn('replay_ids', plan)

    def test_disabled_checkpoint_option_preserves_existing_qualification(self):
        from serve.atomic_summary import model_fingerprint
        cfg = {'exe': __file__, 'atomic_summary_mode': 'state-bank-v1'}
        self.assertEqual(model_fingerprint(cfg), model_fingerprint({**cfg, 'atomic_summary_checkpoint_enabled': False}))
        self.assertNotEqual(model_fingerprint(cfg), model_fingerprint({**cfg, 'atomic_summary_checkpoint_enabled': True}))

    def test_checkpoint_query_owns_the_scheduler_compute_permit(self):
        import contextlib
        import threading
        from types import SimpleNamespace
        from serve.server import Service
        active = []
        owner = object()
        @contextlib.contextmanager
        def control(actual):
            self.assertIs(actual, owner)
            active.append(True)
            try: yield
            finally: active.pop()
        def state(action, **kw):
            self.assertEqual(active, [True])
            self.assertEqual(action, 'checkpoints')
            self.assertIs(kw['binding'], self.main)
            return self.candidates
        svc = Service.__new__(Service)
        svc.engine = SimpleNamespace(summary_checkpoint_enabled=True, max_context=14000,
            atomic_main_snapshot=self.main, summary_state=state)
        svc.summary_lane = SimpleNamespace(scheduler=SimpleNamespace(control=control))
        svc.literals = []
        svc.effort_end = False
        svc.template = SimpleNamespace(render=lambda *a, **kw: 'source')
        svc.tok = SimpleNamespace(encode=lambda text, **kw: [99999] if text == '<|image_pad|>' else self.ids[:4000])
        cancel = threading.Event(); cancel._priority_owner = owner
        plan = svc.summary_checkpoint_plan(dict(atomic_summary_clone='room/turn',
            atomic_summary_checkpoint_source={'messages': [{'role': 'user', 'content': 'source'}]}), self.main, 2000, cancel)
        self.assertEqual(plan['tokens'], 5000)
        self.assertEqual(active, [])

    def test_query_and_fork_bind_both_full_main_and_checkpoint(self):
        engine = ProtocolTests().engine(['SUMMARY_CHECKPOINTS 12000 42 5000 77\n'])
        self.assertEqual(engine.summary_state('checkpoints', binding={'tokens': 12000, 'hash': 42}), [(5000, 77)])
        self.assertEqual(engine.proc.sent, ['SUMMARY_CHECKPOINTS 12000 42\n'])
        engine = ProtocolTests().engine(['SUMMARY_STATE_HELD job1 12000 1000000 1\n'])
        plan = dict(tokens=5000, hash=77, main_snapshot={'tokens': 12000, 'hash': 42})
        held = engine.summary_state('fork', 'job1', max_slot_bytes=2000000, binding=plan)
        self.assertEqual(held['main_tokens'], 12000)
        self.assertEqual(engine.proc.sent, ['SUMMARY_STATE_CHECKPOINT 2000000 job1 12000 42 5000 77\n'])
        engine = ProtocolTests().engine(['SUMMARY_CHECKPOINTS_REFUSED stale-or-busy\n'])
        self.assertEqual(engine.summary_state('checkpoints', binding={'tokens': 12000, 'hash': 42}), [])

if __name__ == '__main__': unittest.main()
