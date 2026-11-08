"""CPU-only tests of negotiated per-turn summary input budgets."""
import threading
import unittest
from serve.atomic_summary import SummaryLane
from serve.priority import WeightedScheduler, PriorityError, Owner
from serve.priority_client import ContinuationClient
from serve.test_priority_client import Engine
from serve.test_priority_scheduler import exhaust

class InputBudget(unittest.TestCase):
    def fixture(self, foreground=False, modern=True):
        engine=Engine();scheduler=WeightedScheduler(threading.Lock());lane=SummaryLane()
        if foreground:exhaust(scheduler.admit(False,threading.Event()))
        cancel=threading.Event();owner=exhaust(scheduler.admit(True,cancel));cancel._priority_owner=owner
        client=ContinuationClient(engine,scheduler,lane)
        if modern:
            send=engine._send_for
            def modern_send(generation,command):
                if command=='ASYNC_CAPS':
                    engine.sent.append(command);engine.lines.put('ASYNC_CAPS 2 512 128 text')
                else:send(generation,command)
            engine._send_for=modern_send
        client.discover()
        return engine,scheduler,client,cancel,owner
    def test_idle_summary_negotiates_128_input_turn(self):
        e,s,c,cancel,owner=self.fixture()
        output=list(c.generate([1,2,3],2,{},cancel))
        self.assertEqual(output,[101,102])
        steps=[x.split() for x in e.sent if x.startswith('ANEXT')]
        self.assertEqual([x[-1] for x in steps],['128','128'])
        self.assertIsNone(s.turn_owner)
    def test_foreground_owner_keeps_summary_preemptible(self):
        e,s,c,cancel,owner=self.fixture(foreground=True)
        self.assertEqual(list(c.generate([1,2,3],2,{},cancel)),[101,102])
        self.assertEqual([x.split()[-1] for x in e.sent if x.startswith('ANEXT')],['1','1'])
    def test_old_native_protocol_is_unchanged(self):
        e,s,c,cancel,owner=self.fixture(modern=False)
        self.assertEqual(list(c.generate([1,2,3],2,{},cancel)),[101,102])
        self.assertTrue(all(len(x.split())==2 for x in e.sent if x.startswith('ANEXT')))
    def test_arrival_changes_next_quantum_without_cancelling_summary(self):
        e,s,c,cancel,bg=self.fixture()
        release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),128);release()
        fg=exhaust(s.admit(False,threading.Event()))
        release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),1);release()
        s.close(fg)
        release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),128);release()
        self.assertFalse(cancel.is_set());self.assertFalse(bg.closed)
    def test_waiting_foreground_reduces_budget(self):
        e,s,c,cancel,bg=self.fixture()
        s.waiters[False].append(Owner(False,threading.Event()))
        release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),1);release()
    def test_budget_cannot_be_chosen_without_compute_ownership(self):
        e,s,c,cancel,bg=self.fixture()
        with self.assertRaises(PriorityError):s.summary_input_quantum(bg)

if __name__=='__main__':unittest.main()
