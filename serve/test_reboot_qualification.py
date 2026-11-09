import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from serve.atomic_summary import model_fingerprint, qualified_runtime_pins, qualification_mismatch

class RebootQualification(unittest.TestCase):
    def test_mount_device_changes_do_not_change_persistent_pins(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);exe=p/'engine';exe.write_bytes(b'verified native code')
            pack=p/'pack';pack.mkdir();(pack/'tokenizer.json').write_text('{"token":1}')
            cfg={'exe':str(exe),'tokenizer':str(pack),'args':[]}
            expected=qualified_runtime_pins(cfg)
            original=Path.stat
            class Remounted:
                def __init__(self, stat):self.stat=stat
                @property
                def st_dev(self):return self.stat.st_dev+10000
                def __getattr__(self,name):return getattr(self.stat,name)
            def remount(path,*args,**kwargs):return Remounted(original(path,*args,**kwargs))
            with patch.object(Path,'stat',remount):
                self.assertEqual(expected,qualified_runtime_pins(cfg))
                self.assertIsNone(qualification_mismatch(dict(cfg,**expected)))
            self.assertTrue(expected['atomic_summary_expected_model_fingerprint'].startswith('strata-stat-v2:'))
            exe.write_bytes(b'changed native code')
            self.assertIsNotNone(qualification_mismatch(dict(cfg,**expected)))
    def test_transient_hash_identity_still_detects_a_device_change(self):
        from serve import atomic_summary as a
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'engine';p.write_bytes(b'native')
            identity=a._file_identity(p)
            info=p.stat()
            changed=SimpleNamespace(st_dev=info.st_dev+1,st_ino=info.st_ino,st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns,st_ctime_ns=info.st_ctime_ns)
            a._stable_file_sha.cache_clear()
            with patch.object(Path,'stat',return_value=changed):
                self.assertNotEqual(identity,a._file_identity(p))
                with self.assertRaisesRegex(ValueError,'changed before hashing'):
                    a._stable_file_sha(str(p),identity)
    def test_artifact_replacement_and_loader_configuration_remain_bound(self):
        with tempfile.TemporaryDirectory() as d:
            exe=Path(d)/'engine';exe.write_bytes(b'native')
            cfg={'exe':str(exe),'args':['--max-context','65536'],'env':{'DYLD_LIBRARY_PATH':'/verified'}}
            first=model_fingerprint(cfg)
            changed=copy.deepcopy(cfg);changed['env']['DYLD_LIBRARY_PATH']='/different'
            self.assertNotEqual(first,model_fingerprint(changed))
            replacement=exe.with_suffix('.new');replacement.write_bytes(b'native');replacement.replace(exe)
            self.assertNotEqual(first,model_fingerprint(cfg))
if __name__=='__main__':unittest.main()
