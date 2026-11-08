import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from serve.disk_session import DiskSession

class Engine:
    def __init__(self): self.calls=[];self.fail=False
    def session_file(self, action, path):
        self.calls.append((action,path))
        if self.fail: raise RuntimeError('save failed')
        if action=='save':Path(path).write_bytes(b'native-session')
        return {'tokens':123,'bytes':14,'ms':1}

class DiskTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cache=DiskSession(self.temp.name,{'model':'one'})
        self.engine=Engine()
    def service(self, **state):
        lane=SimpleNamespace(scheduler=None,status=lambda:{'active':False,**state})
        return SimpleNamespace(loaded=lambda:True,summary_lane=lane,fifo=threading.Lock(),engine=self.engine)
    def test_roundtrip_and_atomic_replacement(self):
        self.assertIsNone(self.cache.restore(self.engine))
        self.cache.save(self.service());first=json.loads(self.cache.manifest.read_text())
        self.cache.save(self.service());second=json.loads(self.cache.manifest.read_text())
        self.assertNotEqual(first['file'],second['file'])
        self.assertFalse((Path(self.temp.name)/first['file']).exists())
        self.assertEqual(self.cache.restore(self.engine)['tokens'],123)
        self.assertEqual((Path(self.temp.name)/second['file']).stat().st_mode & 0o777,0o600)
    def test_failed_save_keeps_previous_published_file(self):
        self.cache.save(self.service());before=self.cache.manifest.read_bytes();self.engine.fail=True
        with self.assertRaises(RuntimeError):self.cache.save(self.service())
        self.assertEqual(self.cache.manifest.read_bytes(),before)
        self.assertEqual(len(list(Path(self.temp.name).glob('*.session'))),1)
    def test_incompatible_identity_never_restores(self):
        self.cache.save(self.service());other=DiskSession(self.temp.name,{'model':'two'})
        before=len(self.engine.calls);self.assertIsNone(other.restore(self.engine));self.assertEqual(len(self.engine.calls),before)
    def test_active_summary_or_foreground_cannot_be_saved(self):
        for key in ['active','pending','rollback_pending','recovery_required','foreground_requests']:
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.cache.save(self.service(**{key:True}))
        self.assertEqual(self.engine.calls,[])
    def test_weighted_owner_prevents_untagged_save(self):
        svc=self.service();svc.summary_lane.scheduler=SimpleNamespace(cv=threading.Condition(),poison_reason=None,turn_owner=None,owners={False:object(),True:None},waiters={False:[],True:[]})
        with self.assertRaises(RuntimeError):self.cache.save(svc)
        self.assertEqual(self.engine.calls,[])
    def test_idle_weighted_scheduler_can_save(self):
        svc=self.service();svc.summary_lane.scheduler=SimpleNamespace(cv=threading.Condition(),poison_reason=None,turn_owner=None,owners={False:None,True:None},waiters={False:[],True:[]})
        self.assertEqual(self.cache.save(svc)['tokens'],123)
    def test_budget_failure_preserves_previous(self):
        self.cache.save(self.service());before=self.cache.manifest.read_bytes();self.cache.max_bytes=1
        with self.assertRaises(ValueError):self.cache.save(self.service())
        self.assertEqual(self.cache.manifest.read_bytes(),before)
    def test_traversal_and_symlink_not_restored(self):
        self.cache.manifest.write_text(json.dumps({'identity':self.cache.identity,'file':'../outside','bytes':14}))
        with self.assertRaises(ValueError):self.cache.restore(self.engine)
        self.assertEqual(self.engine.calls,[])
    def test_changed_file_not_restored(self):
        self.cache.save(self.service());r=json.loads(self.cache.manifest.read_text());(Path(self.temp.name)/r['file']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.cache.restore(self.engine)

if __name__=='__main__':unittest.main()
