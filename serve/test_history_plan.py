import json
from pathlib import Path
import struct
import tempfile
import unittest
from serve.history_plan import retained_match, choose_boundary, token_hash, restored_ids

class Plans(unittest.TestCase):
    def setUp(self):
        self.old=list(range(1000,4000));self.turn=99
        for source in (64,768,1025,1282,1539):self.old[source]=self.turn
        self.snapshot={'owner':'thread/user','ids':self.old}
    def incoming(self, source):return self.old[:64]+self.old[source:2300]+[20]
    def test_all_three_unaligned_phases_refuse_and_an_earlier_aligned_turn_works(self):
        self.assertTrue(retained_match(self.old,self.incoming(768),99))
        for source in (1025,1282,1539):
            self.assertIsNone(retained_match(self.old,self.incoming(source),99))
            result=choose_boundary(self.snapshot,'thread/user',[{'boundary':125,'source':source},{'boundary':123,'source':768}],lambda c:self.incoming(c['source']),99,65536)
            self.assertEqual(result['boundary'],123)
            self.assertEqual(result['source_tokens'],768)
    def test_prefers_nominal_boundary_when_already_aligned(self):
        result=choose_boundary(self.snapshot,'thread/user',[{'boundary':125,'source':768},{'boundary':123,'source':768}],lambda c:self.incoming(c['source']),99,65536)
        self.assertEqual(result['boundary'],125)
    def test_different_owner_or_missing_native_ids_cannot_plan(self):
        for snapshot in (None,{'owner':'other','ids':self.old},{'owner':'thread/user'}):
            self.assertIsNone(choose_boundary(snapshot,'thread/user',[{'boundary':125}],lambda c:self.incoming(768),99,65536)['boundary'])
    def test_budget_and_candidate_order(self):
        self.assertIsNone(choose_boundary(self.snapshot,'thread/user',[{'boundary':125}],lambda c:self.incoming(768),99,5000)['boundary'])
        for candidates in ([],[{'boundary':125},{'boundary':126}],[{'boundary':True}]):
            with self.assertRaises(ValueError):choose_boundary(self.snapshot,'thread/user',candidates,lambda c:self.incoming(1025),99,65536)
    def test_restored_token_table_matches_the_native_commit_hash(self):
        ids=[1,2,3,101];binding={'tokens':len(ids),'hash':token_hash(ids)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'saved.session'
            for version in (1,2):
                header=b'FORMAT00'+struct.pack('<II',version,64)+bytes(48)
                path.write_bytes(header+bytes(18*8+3*8)+struct.pack('<Q',len(ids))+struct.pack('<4i',*ids))
                self.assertEqual(restored_ids(path,binding),ids)
                self.assertIsNone(restored_ids(path,{**binding,'hash':0}))
                self.assertIsNone(restored_ids(path,{**binding,'tokens':5}))
            path.write_bytes(b'bad')
            self.assertIsNone(restored_ids(path,binding))
    def test_http_planning_is_cpu_only_and_does_not_change_the_snapshot(self):
        from serve.test_atomic_summary import Api
        fixture=Api();fixture.setUp()
        try:
            fixture.engine.atomic_main_snapshot={'owner':'thread/user','ids':[1,2,3]}
            before=json.dumps(fixture.engine.atomic_main_snapshot)
            status,body=fixture.call('/v1/atomic/history-plan',{'owner':'thread/user','candidates':[{'boundary':2,'messages':[{'role':'user','content':'History'}]}]})
            self.assertEqual(status,200)
            self.assertIsNone(body['boundary'])
            self.assertEqual(json.dumps(fixture.engine.atomic_main_snapshot),before)
            self.assertEqual(fixture.engine.calls,[])
        finally:fixture.doCleanups()
