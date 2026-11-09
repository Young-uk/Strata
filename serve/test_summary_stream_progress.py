import threading, unittest
from types import SimpleNamespace
from serve.server import prompt_progress
from serve.atomic_summary import request

class SummaryStreamProgress(unittest.TestCase):
    def test_stream_is_explicit_boolean(self):
        body={'atomic_summary':True,'think':False,'messages':[{'role':'user','content':'x'}]}
        self.assertTrue(request(dict(body,stream=True))['stream'])
        for bad in (1,'true',None):
            with self.assertRaises(ValueError):request(dict(body,stream=bad))
    def test_weighted_stream_never_uses_other_lane_counters(self):
        svc=SimpleNamespace(engine=SimpleNamespace(progress=(9000,10000),reused=8000),
            summary_lane=SimpleNamespace(scheduler=object()))
        summary=threading.Event();foreground=threading.Event()
        self.assertIsNone(prompt_progress(svc,summary))
        summary._atomic_input_progress=(128,1000,0,10)
        foreground._atomic_input_progress=(4500,6000,3000,20)
        self.assertEqual(prompt_progress(svc,summary),{'processed':128,'total':1000,'cache':0,'time_ms':10})
        self.assertEqual(prompt_progress(svc,foreground)['processed'],4500)
        self.assertEqual(prompt_progress(svc,summary)['processed'],128)

    def test_real_http_summary_stream_restores_fake_main(self):
        import http.client,json
        from serve.test_atomic_summary import Api,BODY,ENVELOPE
        fixture=Api();fixture.setUp()
        try:
            fixture.svc.summary_lane.configure(fixture.engine,fixture.tmp.name,ENVELOPE)
            conn=http.client.HTTPConnection('127.0.0.1',fixture.httpd.server_address[1],timeout=10)
            try:
                conn.request('POST','/v1/chat/completions',json.dumps(dict(BODY,stream=True,return_progress=True)),
                    {'Content-Type':'application/json','X-Atomic-Owner-Key':'a'*64})
                response=conn.getresponse();body=response.read().decode()
                self.assertEqual(response.status,200,body)
                self.assertIn('text/event-stream',response.getheader('Content-Type'))
                self.assertIn('data: [DONE]',body)
                chunks=[json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: {')]
                text=''.join(chunk.get('choices',[{}])[0].get('delta',{}).get('content','') for chunk in chunks)
                self.assertEqual(text,'summary')
                self.assertEqual(fixture.engine.main,'original')
                self.assertEqual(fixture.engine.calls,['caps','summary_begin','generate','summary_end'])
            finally:conn.close()
        finally:fixture.doCleanups()
