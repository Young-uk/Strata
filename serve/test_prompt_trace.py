import json
import unittest
from serve.prompt_trace import capture, compare, validated

class PromptTraceTests(unittest.TestCase):
    def snap(self, messages=None, tools=None, kwargs=None, ids=None, owner='thread/turn'):
        return capture(messages or [{'role':'user','content':'PRIVATE_SENTINEL'}], tools or [], kwargs or {}, ids or [1,2,3], owner)

    def test_append_is_not_historical_edit(self):
        previous=self.snap()
        current=self.snap(messages=[{'role':'user','content':'PRIVATE_SENTINEL'}, {'role':'assistant','content':'Reply'}, {'role':'user','content':'Next'}], ids=[1,2,3,4,5])
        result=compare(previous,current,[1,2,3,4,5])
        self.assertEqual(result['changes'],['history_appended'])
        self.assertTrue(result['previous_input_prefix_equal'])
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(current))

    def test_history_template_catalog_and_render_changes(self):
        current=self.snap(messages=[{'role':'user','content':'Edited'}], tools=[{'name':'read'}], kwargs={'enable_thinking':False},ids=[9,2,3])
        result=compare(self.snap(),current,[9,2,3])
        self.assertEqual(result['first_changed_message'],0)
        self.assertEqual(set(result['changes']),{'historical_message_changed','tools_changed','template_changed','previous_rendered_input_prefix_changed'})

    def test_same_payload_changed_render(self):
        self.assertEqual(compare(self.snap(),self.snap(ids=[7,2,3]),[7,2,3])['changes'],['previous_rendered_input_prefix_changed'])

    def test_untrusted_persisted_metadata(self):
        for value in ({}, {**self.snap(),'input_tokens':True}, {**self.snap(),'messages':['not-a-hash']}, {**self.snap(),'messages':['a'*64]*2049}):
            self.assertIsNone(validated(value))
        self.assertIsNone(capture([{}]*2049,[],{},[], 'owner'))
        self.assertIsNone(capture([],[],{},[], None))

    def test_no_mutable_alias(self):
        value=self.snap();copy=validated(value);value['messages'].clear()
        self.assertEqual(len(copy['messages']),1)
        self.assertEqual(compare(None,copy,[1,2,3])['changes'],['no_previous_trace'])


class PromptTraceApiTests(unittest.TestCase):
    def test_actual_prepare_and_completion_keep_trace_out_of_prompt(self):
        import io
        import os
        import contextlib
        import urllib.request
        from pathlib import Path
        from unittest import mock
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        from serve.frontend import ChatTemplate
        class Engine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, **kwargs):
                cancel._strata_queue_started()
                try:
                    yield from super().generate(ids,max_new,sampling,cancel,**kwargs)
                finally:
                    if getattr(self,'replace_trace_on_finish',False):
                        self.atomic_disk_owner='newer/turn'
                        self.atomic_prompt_trace={'newer':True}
        tok=ByteTokenizer(); engine=Engine(tok,'OK',max_context=4096)
        svc=Service(engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        httpd=serve(svc,port=0)
        url=f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions'
        def post(text):
            body={'model':'m','messages':[{'role':'user','content':text}],'think':False,'chat_template_kwargs':{'enable_thinking':False},'max_tokens':16,'atomic_cache_owner':'test/turn'}
            request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
        try:
            output=io.StringIO()
            with mock.patch.dict(os.environ,{'STRATA_ATOMIC_PROMPT_TRACE':'1'}),contextlib.redirect_stdout(output):
                a=post('PRIVATE_TRACE_SENTINEL'); first=engine.atomic_prompt_trace
                b=post('PRIVATE_TRACE_EDITED')
            self.assertEqual(a['choices'][0]['message']['content'],'OK')
            self.assertEqual(b['choices'][0]['message']['content'],'OK')
            self.assertIsNotNone(first)
            self.assertIn('historical_message_changed',output.getvalue())
            self.assertNotIn('PRIVATE_TRACE',output.getvalue())
            with mock.patch.dict(os.environ,{'STRATA_ATOMIC_PROMPT_TRACE':'0'}):post('Untraced')
            self.assertIsNone(engine.atomic_prompt_trace)
            engine.replace_trace_on_finish=True
            with mock.patch.dict(os.environ,{'STRATA_ATOMIC_PROMPT_TRACE':'1'}):post('Owner changed')
            self.assertEqual(engine.atomic_prompt_trace,{'newer':True})
        finally:httpd.shutdown();httpd.server_close()
