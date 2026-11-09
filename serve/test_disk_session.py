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
    def test_clone_binding_survives_restart(self):
        binding = {'owner':'thread/turn', 'tokens':123, 'hash':42, 'tail':2, 'history_hash':'c'*64}
        self.engine.atomic_main_snapshot = binding
        self.cache.save(self.service())
        fresh = Engine()
        self.cache.restore(fresh)
        self.assertEqual(fresh.atomic_main_snapshot, {**binding, 'restored':True})

    def test_old_manifest_cannot_inherit_stale_memory_owner(self):
        self.cache.save(self.service())
        self.engine.atomic_main_snapshot = {'owner':'wrong'}
        self.cache.restore(self.engine)
        self.assertIsNone(self.engine.atomic_main_snapshot)

    def test_changed_session_cannot_restore_clone_binding(self):
        self.engine.atomic_main_snapshot = {'owner':'thread/turn', 'tokens':123, 'hash':42, 'tail':2, 'history_hash':'c'*64}
        self.cache.save(self.service())
        record = json.loads(self.cache.manifest.read_text())
        (Path(self.temp.name)/record['file']).write_bytes(b'changed-state!')
        self.cache.restore(self.engine)
        self.assertIsNone(self.engine.atomic_main_snapshot)

    def test_invalid_binding_is_never_published(self):
        for change in ({'tokens':124}, {'hash':-1}, {'tail':2**32}, {'owner':''}, {'history_hash':''}):
            self.engine.atomic_main_snapshot = {'owner':'thread/turn', 'tokens':123, 'hash':42, 'tail':2, 'history_hash':'c'*64, **change}
            self.cache.save(self.service())
            self.assertNotIn('clone_binding', json.loads(self.cache.manifest.read_text()))

    def test_reviewed_backend_migration_preserves_native_and_model_checks(self):
        previous = {'atomic_summary_expected_native_sha256':'native',
                    'atomic_summary_expected_model_fingerprint':'model',
                    'atomic_summary_expected_backend_sha256':'a'*64}
        old = DiskSession(self.temp.name, previous)
        old.save(self.service())
        current = {**previous, 'atomic_summary_expected_backend_sha256':'b'*64}
        migrated = DiskSession(self.temp.name, current, previous_backend_sha256='a'*64)
        self.assertEqual(migrated.restore(self.engine)['tokens'],123)
        for key in ('atomic_summary_expected_native_sha256','atomic_summary_expected_model_fingerprint'):
            refused=DiskSession(self.temp.name,{**current,key:'different'},previous_backend_sha256='a'*64)
            self.assertIsNone(refused.restore(self.engine))

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


class ReceiptApiTests(unittest.TestCase):
    def test_receipt_binds_only_current_owner_without_inference(self):
        from serve.test_atomic_summary import Api
        fixture=Api();fixture.setUp()
        try:
            fixture.engine.atomic_main_snapshot={'owner':'thread/turn','tokens':123,'hash':42,'tail':2}
            self.assertEqual(fixture.call('/v1/atomic/cache-receipt',{'owner':'wrong','history_hash':'c'*64})[0],409)
            self.assertEqual(fixture.call('/v1/atomic/cache-receipt',{'owner':'thread/turn','history_hash':'invalid'})[0],400)
            self.assertEqual(fixture.call('/v1/atomic/cache-receipt',{'owner':'thread/turn','history_hash':'c'*64})[0],200)
            self.assertEqual(fixture.engine.atomic_main_snapshot['history_hash'],'c'*64)
            self.assertEqual(fixture.engine.calls,[])
        finally:
            fixture.doCleanups()

class ThreadOwnershipTests(unittest.TestCase):
    setUp = DiskTests.setUp
    service = DiskTests.service
    def test_owner_is_independent_of_clone_receipts(self):
        self.engine.atomic_disk_owner = 'room-a/turn'
        self.cache.save(self.service())
        record = json.loads(self.cache.manifest.read_text())
        self.assertEqual(record['cache_owner'], 'room-a/turn')
        self.assertNotIn('clone_binding', record)
        self.assertEqual((self.cache.directory / (record['file'] + '.threads')).read_text(), 'room-a\n')
        restored = Engine();self.cache.restore(restored)
        self.assertEqual(restored.atomic_disk_owner, 'room-a/turn')

    def test_deleted_owner_cannot_save_or_restore(self):
        self.engine.atomic_disk_owner = 'room-a/turn'
        self.cache.save(self.service())
        markers=self.cache.directory/'.deleted-threads';markers.mkdir();(markers/'room-a').write_text('deleted')
        calls=len(self.engine.calls)
        self.assertIsNone(self.cache.save(self.service()))
        self.assertIsNone(self.cache.restore(self.engine))
        self.assertEqual(len(self.engine.calls),calls)

    def test_deletion_during_native_save_cannot_resurrect_cache(self):
        self.engine.atomic_disk_owner = 'room-a/turn'
        original=self.engine.session_file
        def delete_during_save(action,path):
            result=original(action,path)
            markers=self.cache.directory/'.deleted-threads';markers.mkdir();(markers/'room-a').write_text('deleted')
            return result
        self.engine.session_file=delete_during_save
        self.assertIsNone(self.cache.save(self.service()))
        self.assertFalse(self.cache.manifest.exists())
        self.assertFalse(list(self.cache.directory.glob('*.session*')))
