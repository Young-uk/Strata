"""CPU-only qualification drift tests; no model/runtime launch."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from serve import atomic_summary as a
class Engine:
    max_context=65536
    batch=0
    session_save_reclaim=False
    def __init__(self):self.calls=[]
    def summary_control(self,action):self.calls.append(action)
    def session_file(self,*args,**kwargs):raise AssertionError('Must refuse before native mutation')
class Binding(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.native=self.root/'engine';self.native.write_bytes(b'fake-native')
        self.pack=self.root/'pack';self.pack.mkdir();(self.pack/'dense.bin').write_bytes(b'fake-dense')
        self.cfg={'exe':str(self.native),'cwd':str(self.root),'args':['--pack',str(self.pack),'--max-context','65536'],
            'atomic_summary_transaction_qualified':True,'atomic_completion_review_qualified':False,
            'atomic_summary_qualified_max_main_tokens':2048,'atomic_summary_qualified_max_snapshot_bytes':256<<20,
            'atomic_summary_qualified_max_protected_bytes':1<<30,'atomic_summary_snapshot_path':str(self.root/'snapshots')}
    def pin(self):self.cfg.update(a.qualified_runtime_pins(self.cfg))
    def configure(self,lane,engine):
        if not hasattr(engine,'spawn'):
            engine.spawn=(str(self.native),list(self.cfg['args']),str(self.root),None,{})
        return a.configure_optional_lane(lane,engine,self.cfg,engine_name='strata',arch='qwen4exp')
    def test_pins_nonrecursive(self):
        self.pin();self.assertIsNone(a.qualification_mismatch(self.cfg))
        self.assertEqual({k:self.cfg[k] for k in a.QUALIFICATION_PIN_KEYS},a.qualified_runtime_pins(self.cfg))
    def test_missing_pins_keep_foreground_available(self):
        lane=a.SummaryLane();engine=Engine();self.assertFalse(self.configure(lane,engine));self.assertFalse(lane.enabled)
        self.assertIn('missing_or_invalid',lane.status()['disabled_reason']);lane.check_healthy()
        with lane.register(False,threading.Event()):self.assertEqual(lane.foreground,1)
        self.assertEqual(engine.calls,[])
    def test_each_wrong_pin(self):
        self.pin()
        for key in a.QUALIFICATION_PIN_KEYS:
            with self.subTest(key=key):
                cfg=dict(self.cfg);cfg[key]=('strata-stat-v2:' if key.endswith('fingerprint') else '')+'0'*64
                self.assertTrue(a.qualification_mismatch(cfg).startswith('changed_'))
    def test_native_change(self):
        self.pin();self.native.write_bytes(b'new-native');self.assertIn('native',a.qualification_mismatch(self.cfg))
    def test_pack_inplace_change(self):
        self.pin();(self.pack/'dense.bin').write_bytes(b'new-dense');self.assertIn('model_fingerprint',a.qualification_mismatch(self.cfg))
    def test_config_and_environment_change(self):
        self.pin();changed={**self.cfg,'args':self.cfg['args']+['--prefill','1024']}
        self.assertIn('model_fingerprint',a.qualification_mismatch(changed))
        with patch.dict('os.environ',{'STRATA_TEST_QUALIFICATION_DRIFT':'1'}):self.assertIn('model_fingerprint',a.qualification_mismatch(self.cfg))
    def test_bad_envelope_and_caps_nonfatal(self):
        self.cfg['atomic_summary_qualified_max_main_tokens']=0;self.pin();lane=a.SummaryLane()
        self.assertFalse(self.configure(lane,Engine()));lane.check_healthy()
        self.cfg['atomic_summary_qualified_max_main_tokens']=2048;self.pin();engine=Engine()
        with patch.object(engine,'summary_control',side_effect=RuntimeError('caps unavailable')):
            self.assertFalse(self.configure(lane,engine));lane.check_healthy()
    def test_observation_error(self):
        self.pin();self.native.unlink();lane=a.SummaryLane();self.assertFalse(self.configure(lane,Engine()))
        self.assertEqual(lane.disabled_reason,'qualification_identity_unavailable')
    def test_each_begin_rechecks_before_native_without_poison(self):
        self.pin();lane=a.SummaryLane();engine=Engine();self.assertTrue(self.configure(lane,engine))
        (self.pack/'dense.bin').write_bytes(b'changed')
        with self.assertRaises(a.SummaryError) as raised:
            with lane.transaction(engine,threading.Event()):self.fail('must not generate')
        self.assertEqual(raised.exception.code,'atomic_summary_admission_refused');self.assertFalse(lane.enabled)
        self.assertFalse(lane.poisoned);self.assertIsNone(lane.active);self.assertEqual(engine.calls,['caps']);lane.check_healthy()
    def test_backend_digest_change(self):
        self.pin();original=a.qualified_runtime_pins
        def changed(cfg):
            result=original(cfg);result['atomic_summary_expected_backend_sha256']='0'*64;return result
        with patch.object(a,'qualified_runtime_pins',side_effect=changed):self.assertIn('backend',a.qualification_mismatch(self.cfg))
    def test_actual_spawn_overrides_stale_config(self):
        self.pin()
        for slot,value in [(0,str(self.root/'other-native')),(1,self.cfg['args']+['--max-context','32768']),(4,{'STRATA_METAL_F16':'1'})]:
            engine=Engine();spawn=[str(self.native),list(self.cfg['args']),str(self.root),None,{}];spawn[slot]=value;engine.spawn=tuple(spawn)
            lane=a.SummaryLane();self.assertFalse(self.configure(lane,engine));self.assertFalse(lane.poisoned);lane.check_healthy()
    def test_user_override_changes_fingerprint_even_if_original_args_same(self):
        self.pin();changed={**self.cfg,'user_args':['--max-context','32768']}
        self.assertIn('model_fingerprint',a.qualification_mismatch(changed))
    def test_spawn_change_after_configuration_refuses_begin(self):
        self.pin();lane=a.SummaryLane();engine=Engine();self.assertTrue(self.configure(lane,engine))
        engine.spawn=(str(self.native),self.cfg['args']+['--prefill','128'],str(self.root),None,{})
        with self.assertRaises(a.SummaryError):
            with lane.transaction(engine,threading.Event()):self.fail('must not generate')
        self.assertEqual(engine.calls,['caps']);lane.check_healthy()
    def test_prelaunch_helper_observes_resolved_args_and_env_without_starting(self):
        from serve import server
        actual_args=self.cfg['args']+['--prefill','128']
        with patch.object(server,'apply_saved_tuning',side_effect=lambda cfg:cfg), patch.object(server,'with_component_engine',side_effect=lambda cfg:cfg), patch.object(server,'apply_vision_defaults',side_effect=lambda cfg:cfg), patch.object(server,'engine_args',return_value=actual_args), patch.object(server,'child_env',return_value={'STRATA_METAL_F16':'0'}), patch.object(server,'configure_batch_telemetry_env'), patch.object(server,'StrataEngine',side_effect=AssertionError('no launch')):
            pins=server.prepare_summary_qualification_pins(self.cfg)
        engine=Engine();engine.spawn=(str(self.native),actual_args,str(self.root),None,{'STRATA_METAL_F16':'0'})
        self.cfg.update(pins);self.assertTrue(self.configure(a.SummaryLane(),engine))
    def test_mock_startup_reaches_http_boundary_without_strata_locals(self):
        from serve import server
        class ReachedHTTP(Exception):pass
        with patch.object(server.lifeline,'install_stop_handlers'), patch.object(server,'Server'), patch.object(server,'serve',side_effect=ReachedHTTP) as serve:
            with self.assertRaises(ReachedHTTP):server._main(['--engine','mock','--port','0','--tokenizer',str(self.root/'absent-tokenizer')])
            svc=serve.call_args.args[0]
            self.assertFalse(svc.summary_lane.enabled);self.assertFalse(svc.atomic_completion_review_qualified)

if __name__=='__main__':unittest.main()
