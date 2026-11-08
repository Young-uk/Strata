"""CPU-only protocol/state-machine tests. No model or live service."""
import threading
import unittest
from unittest.mock import patch
from serve.atomic_summary import SummaryLane, SummaryError
from serve.server import EngineDied, SessionRefused
from serve.test_slots import FakeEngine

class Bank:
    max_context=65536
    batch=0
    session_save_reclaim=False
    def __init__(self): self.calls=[];self.fail=None;self.main={'image':'retained','cache':[1,2],'tokens':53000}
    def summary_state(self,action,txid=None,**kw):
        self.calls.append(action)
        if action=='caps':return True
        if self.fail==action:raise EngineDied('uncertain')
        if self.fail=='refuse' and action=='begin':
            e=SessionRefused('memory','pressure');e.summary_refused=True;raise e
        return {'transaction_id':txid,'main_tokens':53000,'allocated_bytes':100,'epoch':'2' if self.fail=='mismatch' and action=='end' else '1'}

class LaneTests(unittest.TestCase):
    def setUp(self):self.engine=Bank();self.lane=SummaryLane();self.lane.configure_state_bank(self.engine,1024)
    def test_no_disk_and_main_unchanged(self):
        before=repr(self.engine.main)
        with patch('serve.atomic_summary.tempfile.mkstemp',side_effect=AssertionError('disk forbidden')):
            with self.lane.transaction(self.engine,threading.Event()):pass
        self.assertEqual(repr(self.engine.main),before);self.assertEqual(self.engine.calls,['caps','begin','end'])
        self.assertIsNone(self.lane.status()['qualified_envelope'])
    def test_clean_refusal_does_not_poison(self):
        self.engine.fail='refuse'
        with self.assertRaises(SummaryError) as e:
            with self.lane.transaction(self.engine,threading.Event()):pass
        self.assertEqual(e.exception.code,'atomic_summary_admission_refused');self.assertFalse(self.lane.poisoned)
    def test_uncertain_begin_and_end_poison(self):
        for mode in ('begin','end','mismatch'):
            engine=Bank();engine.fail=mode;lane=SummaryLane();lane.configure_state_bank(engine,1024)
            with self.assertRaises(SummaryError):
                with lane.transaction(engine,threading.Event()):pass
            self.assertTrue(lane.poisoned);self.assertIsNone(lane.recovery_path)
    def test_cancel_and_foreground_restore_first(self):
        cancel=threading.Event()
        with self.assertRaises(SummaryError) as e:
            with self.lane.transaction(self.engine,cancel):self.lane.foreground_arrived()
        self.assertEqual(e.exception.code,'atomic_summary_interrupted');self.assertEqual(self.engine.calls[-1],'end');self.assertFalse(self.lane.poisoned)
    def test_cancel_before_begin(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(SummaryError):
            with self.lane.transaction(self.engine,cancel):pass
        self.assertEqual(self.engine.calls,['caps'])
    def test_binding_drift_disables_only_optional(self):
        self.lane.binding_check=lambda:'changed_pin'
        with self.assertRaises(SummaryError):
            with self.lane.transaction(self.engine,threading.Event()):pass
        self.assertFalse(self.lane.enabled);self.assertFalse(self.lane.poisoned)
    def test_bad_budget(self):
        for value in (None,True,0,-1,9007199254740992):
            with self.assertRaises(ValueError):SummaryLane().configure_state_bank(self.engine,value)

class ProtocolTests(unittest.TestCase):
    def engine(self,lines):
        e=FakeEngine();e.max_context=65536;e.script=lines;return e
    def test_caps(self):self.assertTrue(self.engine(['SUMMARY_STATE_CAPS 4 16384\n']).summary_state('caps'))
    def test_no_protocol_fallback(self):
        for text in ('SUMMARY_CAPS 3 16384','SUMMARY_STATE_CAPS 0 16384','SUMMARY_STATE_CAPS 4 8192'):
            with self.assertRaises(ValueError):self.engine([text+'\n']).summary_state('caps')
    def test_roundtrip(self):
        e=self.engine(['SUMMARY_STATE_HELD abc 53000 100 1\n']);held=e.summary_state('begin','abc',max_slot_bytes=1024)
        self.assertEqual(e.proc.sent[-1],'SUMMARY_STATE_BEGIN 1024 abc\n')
        e.script=['SUMMARY_STATE_RELEASED abc 53000 100 1\n'];self.assertEqual(held,e.summary_state('end','abc'))
    def test_refusal_only_for_matching_begin(self):
        e=self.engine(['SUMMARY_STATE_REFUSED abc pressure\n'])
        with self.assertRaises(SessionRefused) as ex:e.summary_state('begin','abc',max_slot_bytes=1024)
        self.assertTrue(ex.exception.summary_refused)
        for action in ('begin','end'):
            e=self.engine(['SUMMARY_STATE_REFUSED wrong pressure\n'])
            with self.assertRaises(EngineDied):e.summary_state(action,'abc',max_slot_bytes=1024)
    def test_malformed_ack(self):
        for text in ('SUMMARY_STATE_HELD wrong 1 2 3','SUMMARY_STATE_HELD abc -1 2 3','SUMMARY_STATE_HELD abc 1 2048 3','SUMMARY_STATE_HELD abc 1 2 0','SUMMARY_STATE_HELD abc 1 2 01','SUMMARY_STATE_HELD abc 1 2 3 extra'):
            with self.assertRaises(EngineDied):self.engine([text+'\n']).summary_state('begin','abc',max_slot_bytes=1024)
    def test_injection_rejected_before_send(self):
        e=self.engine([])
        for tx in ('','a b','a\nQUIT','x'*65):
            with self.assertRaises(ValueError):e.summary_state('end',tx)
        self.assertEqual(e.proc.sent,[])
class ConfigTests(unittest.TestCase):
    def test_explicit_mode_provenance_and_pins_required(self):
        from serve.atomic_summary import configure_optional_lane
        cfg={'atomic_summary_mode':'state-bank-v1','atomic_summary_state_bank_enabled':True,
             'atomic_summary_transaction_qualified':False,'atomic_summary_state_bank_max_bytes':1024,
             'atomic_summary_state_bank_provenance':'experimental-user-validation'}
        engine=Bank()
        with patch('serve.atomic_summary.runtime_qualification_config',side_effect=lambda c,e:c), patch('serve.atomic_summary.qualification_mismatch',return_value=None):
            lane=SummaryLane();self.assertTrue(configure_optional_lane(lane,engine,cfg,engine_name='strata',arch='qwen4exp'))
            self.assertEqual(lane.mode,'state-bank-v1');self.assertIsNone(lane.envelope)
            for changes in ({'atomic_summary_transaction_qualified':True},{'atomic_summary_state_bank_provenance':'qualified'},{'atomic_summary_state_bank_max_bytes':0}):
                self.assertFalse(configure_optional_lane(SummaryLane(),engine,{**cfg,**changes},engine_name='strata',arch='qwen4exp'))
        with patch('serve.atomic_summary.runtime_qualification_config',side_effect=lambda c,e:c), patch('serve.atomic_summary.qualification_mismatch',return_value='changed_pin'):
            self.assertFalse(configure_optional_lane(SummaryLane(),engine,cfg,engine_name='strata',arch='qwen4exp'))

class Capabilities(unittest.TestCase):
    def test_experimental_review_transport_does_not_claim_qualification(self):
        from serve.test_atomic_summary import Api
        from serve.priority import WeightedScheduler
        fixture = Api(); fixture.setUp()
        try:
            lane = fixture.svc.summary_lane
            def advertised(expected):
                status, body = fixture.call('/v1/models')
                self.assertEqual(status, 200)
                self.assertIs(body['atomic_completion_review_experimental'], expected)
                self.assertIs(body['atomic_completion_review'], False)
            advertised(False)  # No enabled isolated transport.
            fixture.engine.summary_state = Bank().summary_state
            lane.configure_state_bank(fixture.engine, 1024)
            advertised(True)  # Serial state-bank transport is also experimental.
            lane.scheduler = WeightedScheduler(fixture.svc.fifo, 8)
            advertised(True)
            lane.enabled = False
            advertised(False)
            lane.enabled = True
            lane.poisoned = True
            advertised(False)
            lane.poisoned = False
            lane.scheduler.poison('test recovery required')
            advertised(False)
            lane.scheduler = None
            # Even a mistaken semantic config flag cannot qualify state-bank review.
            fixture.svc.atomic_completion_review_qualified = True
            advertised(True)
            lane.mode = 'serial-snapshot-v1'
            fixture.svc.atomic_completion_review_qualified = False
            advertised(False)
        finally:
            fixture.doCleanups()

    def test_public_bank_contract(self):
        from serve.test_atomic_summary import Api
        fixture=Api();fixture.setUp()
        try:
            fixture.engine.summary_state=Bank().summary_state
            fixture.svc.summary_lane.configure_state_bank(fixture.engine,1024)
            fixture.svc.atomic_completion_review_qualified=True
            status,body=fixture.call('/v1/models')
            self.assertEqual(status,200);self.assertTrue(body['atomic_summary_slot'])
            self.assertEqual(body['atomic_summary_protocol'],4)
            self.assertEqual(body['atomic_summary_context'],16384)
            self.assertEqual(body['atomic_summary_isolation'],'shared-weights-state-bank-v1')
            self.assertEqual(body['atomic_summary_state_bank_provenance'],'experimental-user-validation')
            for key in ('atomic_summary_shared_weights','atomic_summary_serialized_compute','atomic_summary_experimental','atomic_summary_requires_foreground_warmup'):self.assertTrue(body[key])
            self.assertFalse(body['atomic_summary_cold_start']);self.assertFalse(body['atomic_completion_review'])
            self.assertIsNone(body['atomic_summary_qualified_envelope'])
        finally:fixture.doCleanups()

class ServiceSafety(unittest.TestCase):
    def test_poison_blocks_explicit_load_and_restart_before_hooks(self):
        from serve.server import Service, ByteTokenizer, MockEngine
        from serve.frontend import ChatTemplate
        from serve.test_atomic_summary import ROOT
        tok=ByteTokenizer();engine=MockEngine(tok,'answer');svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        svc.summary_lane.poisoned=True
        with patch.object(svc,'loaded',return_value=False), patch.object(engine,'restart',create=True) as restart, patch('serve.server.subprocess.run') as hook:
            with self.assertRaises(SummaryError):svc.ensure_loaded()
            self.assertFalse(restart.called);self.assertFalse(hook.called)
            with self.assertRaises(SummaryError):list(svc.run([1],False,[],1,{},threading.Event()))
            self.assertFalse(restart.called)

    def test_foreground_waits_for_summary_drain_and_end(self):
        from serve.server import Service, ByteTokenizer, MockEngine
        from serve.frontend import ChatTemplate
        from serve.test_atomic_summary import ROOT
        class Engine(MockEngine):
            def __init__(self,tok):
                super().__init__(tok,'foreground',max_context=65536)
                self.started=threading.Event();self.events=[];self.bank=False
            def summary_state(self,action,txid=None,**kw):
                if action=='caps':return True
                self.events.append(action)
                if action=='begin':self.bank=True
                else:
                    assert 'drained' in self.events
                    self.bank=False
                return {'transaction_id':txid,'main_tokens':53000,'allocated_bytes':100,'epoch':'1'}
            def generate(self,ids,max_new,sampling,cancel,**kw):
                if sampling.get('atomic_summary'):
                    self.started.set()
                    try:
                        if not cancel.wait(3):raise AssertionError('foreground did not preempt summary')
                        return
                        yield 0
                    finally:self.events.append('drained')
                assert not self.bank
                self.events.append('foreground')
                yield from super().generate(ids,max_new,sampling,cancel,**kw)
        tok=ByteTokenizer();engine=Engine(tok);svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        svc.summary_lane.configure_state_bank(engine,1024)
        results=[]
        def summary():
            try:results.extend(svc.run([1],False,[],8,{'atomic_summary':True},threading.Event()))
            except SummaryError as exc:results.append(exc.code)
        thread=threading.Thread(target=summary);thread.start()
        self.assertTrue(engine.started.wait(2))
        foreground=list(svc.run([2],False,[],8,{},threading.Event()))
        thread.join(4);self.assertFalse(thread.is_alive())
        self.assertEqual(engine.events,['begin','drained','end','foreground'])
        self.assertIn('atomic_summary_interrupted',results)
        self.assertTrue(any(kind=='done' for kind,_ in foreground));self.assertFalse(svc.summary_lane.poisoned)

if __name__=='__main__':unittest.main()
