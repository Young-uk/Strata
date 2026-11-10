"""CPU/fake-engine coverage; never loads model weights or touches a live server."""
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.atomic_summary import SummaryLane, SummaryError, request
from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, serve, SessionRefused

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = {"max_main_tokens": 8192, "max_snapshot_bytes": 64 << 20, "max_protected_bytes": 128 << 20}
BODY = {"atomic_summary": True, "think": False, "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32}


class Fake(MockEngine):
    def __init__(self, tok):
        super().__init__(tok, "summary", max_context=65536)
        self.calls = []
        self.fail = None
        self.main = "original"
        self.original = None
        self.main_tokens = 12
        self.snapshot_bytes = 8
        self.protected_bytes = 16
        self.last_envelope = None

    def summary_control(self, action):
        self.calls.append(action)
        return True

    def session_file(self, action, path, **kw):
        self.calls.append(action)
        if action == self.fail:
            raise RuntimeError("lost protocol response")
        if action == "summary_begin":
            self.last_envelope = kw["summary_envelope"]
            if (self.main_tokens > self.last_envelope["max_main_tokens"] or
                    self.snapshot_bytes > self.last_envelope["max_snapshot_bytes"] or
                    self.protected_bytes > self.last_envelope["max_protected_bytes"]):
                refused = SessionRefused("memory", "outside measured qualification envelope", False)
                refused.summary_refused = True
                raise refused
            self.original = self.main
            Path(path).write_text("snapshot")
        else:
            self.main = self.original
        return {"tokens": 12, "bytes": 8, "ms": 1}

    def generate(self, *args, **kw):
        self.calls.append("generate")
        self.main = "summary"
        yield from super().generate(*args, **kw)


class Lane(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = Fake(ByteTokenizer())
        self.lane = SummaryLane()
        self.lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.cancel = threading.Event()

    def run_transaction(self, action=lambda: None):
        with self.lane.register(True, self.cancel):
            with self.lane.transaction(self.engine, self.cancel):
                action()

    def test_success_restores_and_removes_snapshot(self):
        self.run_transaction(lambda: setattr(self.engine, "main", "summary"))
        self.assertEqual(self.engine.main, "original")
        self.assertEqual(self.engine.calls, ["caps", "summary_begin", "summary_end"])
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_foreground_cancels_then_restore_completes(self):
        def foreground():
            with self.lane.register(False, threading.Event()):
                self.assertTrue(self.cancel.is_set())
        with self.assertRaises(SummaryError) as got:
            self.run_transaction(foreground)
        self.assertEqual(got.exception.status, 409)
        self.assertEqual(self.engine.calls[-1], "summary_end")
        self.assertFalse(self.lane.poisoned)

    def test_foreground_waiter_prevents_begin(self):
        with self.lane.register(False, threading.Event()):
            with self.assertRaises(SummaryError):
                self.run_transaction()
        self.assertEqual(self.engine.calls, ["caps"])

    def test_generation_failure_still_restores(self):
        def fail():
            raise ValueError("bad generation")
        with self.assertRaises(ValueError):
            self.run_transaction(fail)
        self.assertEqual(self.engine.calls[-1], "summary_end")
        self.assertFalse(self.lane.poisoned)

    def test_end_failure_blocks_foreground_and_keeps_snapshot(self):
        self.engine.fail = "summary_end"
        with self.assertRaises(SummaryError):
            self.run_transaction()
        self.assertTrue(Path(self.lane.recovery_path).exists())
        with self.assertRaises(SummaryError):
            with self.lane.register(False, threading.Event()):
                self.fail("must never admit")

    def test_uncertain_begin_failure_is_poisoned(self):
        self.engine.fail = "summary_begin"
        with self.assertRaises(RuntimeError):
            self.run_transaction()
        self.assertTrue(self.lane.poisoned)
        self.assertEqual(self.engine.calls, ["caps", "summary_begin"])

    def test_structured_begin_refusal_does_not_poison(self):
        refusal = SessionRefused("memory", "not enough RAM", False)
        refusal.summary_refused = True
        with patch.object(self.engine, "session_file", side_effect=refusal):
            with self.assertRaises(SummaryError):
                self.run_transaction()
        self.assertFalse(self.lane.poisoned)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def empty_protocol(self, *, missing=False, fail_end=False):
        self.engine.main = ""
        def session(action, path, **kw):
            self.engine.calls.append(action)
            if action == "summary_begin":
                if missing:
                    Path(path).unlink()
            elif fail_end:
                raise RuntimeError("empty reset failed")
            else:
                self.engine.main = ""
            return {"tokens": 0, "bytes": 0, "ms": 0.1}
        return patch.object(self.engine, "session_file", side_effect=session)

    def test_pristine_empty_success_accepts_placeholder_or_no_file(self):
        for missing in (False, True):
            with self.subTest(missing=missing), self.empty_protocol(missing=missing):
                def generate():
                    self.assertEqual(self.lane.status()["transaction_mode"], "empty")
                    self.engine.main = "summary"
                self.run_transaction(generate)
                self.assertEqual(self.engine.main, "")
                self.assertFalse(self.lane.poisoned)
                self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_pristine_empty_cancellation_still_resets(self):
        with self.empty_protocol(missing=True):
            with self.assertRaises(SummaryError) as got:
                self.run_transaction(self.lane.foreground_arrived)
        self.assertEqual(got.exception.status, 409)
        self.assertEqual(self.engine.main, "")
        self.assertFalse(self.lane.poisoned)
        self.assertEqual(self.engine.calls[-1], "summary_end")

    def test_empty_rollback_failure_does_not_claim_saved_main_kv(self):
        with self.empty_protocol(missing=True, fail_end=True):
            with self.assertRaises(SummaryError) as got:
                self.run_transaction()
        self.assertTrue(self.lane.poisoned)
        self.assertEqual(self.lane.status()["transaction_mode"], "empty")
        self.assertIn("no saved main KV existed", str(got.exception))
        self.assertFalse(Path(self.lane.recovery_path).exists())

    def test_disk_snapshot_missing_is_not_interpreted_as_empty(self):
        def begin(action, path, **kw):
            Path(path).unlink()
            return {"tokens": 12, "bytes": 8, "ms": 1}
        with patch.object(self.engine, "session_file", side_effect=begin):
            with self.assertRaises(SummaryError):
                self.run_transaction(lambda: self.fail("must not generate"))
        self.assertTrue(self.lane.poisoned)
        self.assertEqual(self.lane.status()["transaction_mode"], "unknown")

    def test_rollback_counts_must_match_begin(self):
        original = self.engine.session_file
        def protocol(action, path, **kw):
            result = original(action, path, **kw)
            return {**result, "tokens": 13} if action == "summary_end" else result
        with patch.object(self.engine, "session_file", side_effect=protocol):
            with self.assertRaises(SummaryError):
                self.run_transaction()
        self.assertTrue(self.lane.poisoned)

    def test_mixed_empty_disk_counts_fail_before_generation(self):
        with patch.object(self.engine, "session_file", return_value={"tokens": 0, "bytes": 8, "ms": 1}):
            with self.assertRaises(SummaryError):
                self.run_transaction(lambda: self.fail("must not generate"))
        self.assertTrue(self.lane.poisoned)

    def test_temporary_file_error_clears_active(self):
        with patch("serve.atomic_summary.tempfile.mkstemp", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_transaction()
        self.assertIsNone(self.lane.active)
        self.assertFalse(self.lane.poisoned)


class Api(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = Fake(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.atomic_owner_key = "a" * 64
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def call(self, path, body=None, owner=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            conn.request("GET" if body is None else "POST", path,
                         None if body is None else json.dumps(body), {"Content-Type": "application/json",
                         **({"X-Atomic-Owner-Key": "a" * 64} if owner else {})})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def test_disabled_never_generates(self):
        status, body = self.call("/v1/chat/completions", BODY)
        self.assertEqual(status, 501)
        self.assertEqual(self.engine.calls, [])
        status, body = self.call("/v1/models")
        self.assertFalse(body["atomic_summary_slot"])
        self.assertFalse(body["atomic_summary_cold_start"])
        self.assertTrue(body["atomic_summary_token_count"])
        self.assertEqual(body["runtime_id"], body["data"][0]["runtime_id"])
        self.assertEqual(body["atomic_tool_preamble"], 0)

    def test_completion_review_requires_own_qualification_and_ready_lane(self):
        self.assertFalse(self.call("/v1/models")[1]["atomic_completion_review"])
        self.svc.atomic_completion_review_qualified = True
        self.assertFalse(self.call("/v1/models")[1]["atomic_completion_review"])
        # Review wire shape still refuses when the shared isolated lane is absent.
        self.assertEqual(self.call("/v1/chat/completions", {**BODY, "max_tokens": 64})[0], 501)
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.assertTrue(self.call("/v1/models")[1]["atomic_completion_review"])
        self.svc.atomic_completion_review_qualified = False
        caps = self.call("/v1/models")[1]
        self.assertTrue(caps["atomic_summary_slot"])
        self.assertFalse(caps["atomic_completion_review"])
        self.svc.atomic_completion_review_qualified = "true"
        self.assertFalse(self.call("/v1/models")[1]["atomic_completion_review"])
        self.svc.atomic_completion_review_qualified = True
        self.svc.summary_lane.poisoned = True
        self.assertFalse(self.call("/v1/models")[1]["atomic_completion_review"])

    def test_completion_review_false_after_native_discovery_failure(self):
        self.svc.atomic_completion_review_qualified = True
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.assertTrue(self.call("/v1/models")[1]["atomic_completion_review"])
        with patch.object(self.engine, "summary_control", side_effect=ValueError("SUMMARY_CAPS 0 16384")):
            with self.assertRaises(ValueError):
                self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        caps = self.call("/v1/models")[1]
        self.assertFalse(caps["atomic_summary_slot"])
        self.assertFalse(caps["atomic_completion_review"])
        self.assertEqual(self.call("/v1/chat/completions", {**BODY, "max_tokens": 64})[0], 501)

    def test_qualified_envelope_is_advertised_without_reducing_context(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        caps = self.call("/v1/models")[1]
        self.assertEqual(caps["atomic_summary_qualified_envelope"], ENVELOPE)
        self.assertEqual(caps["data"][0]["context_length"], 65536)
        self.assertEqual(caps["atomic_summary_context"], 16384)
        self.assertEqual(caps["atomic_summary_protocol"], 3)
        self.assertTrue(caps["atomic_summary_cold_start"])

    def test_larger_main_state_refuses_summary_before_any_generation(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        for attribute, value in (("main_tokens", 8193), ("snapshot_bytes", 64 * 1024 * 1024 + 1),
                                 ("protected_bytes", 128 * 1024 * 1024 + 1)):
            with self.subTest(bound=attribute):
                self.engine.main_tokens, self.engine.snapshot_bytes, self.engine.protected_bytes = 12, 8, 16
                setattr(self.engine, attribute, value)
                status, body = self.call("/v1/chat/completions", BODY)
                self.assertEqual(status, 409, body)
                self.assertEqual(body["error"]["code"], "atomic_summary_admission_refused")
                self.assertEqual(self.engine.last_envelope, ENVELOPE)
                self.assertEqual(self.engine.main, "original")
                self.assertNotIn("generate", self.engine.calls)
                self.assertFalse(self.svc.summary_lane.poisoned)

    def test_cold_recovery_rejects_stale_owner_and_mixed_clone_before_generation(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.engine.atomic_main_snapshot = {"owner":"thread/current", "tokens":12, "hash":7, "ids":list(range(12)), "tail":4}
        payload = {**BODY, "atomic_summary_cold_recovery":"thread/old", "atomic_summary_recovery":True}
        status, body = self.call('/v1/chat/completions', payload)
        self.assertEqual(status, 409, body)
        self.assertEqual(body['error']['code'], 'atomic_summary_clone_stale')
        status, body = self.call('/v1/chat/completions', {**payload, "atomic_summary_clone":"thread/current"})
        self.assertEqual(status, 400, body)
        self.assertNotIn('generate', self.engine.calls)
        self.assertNotIn('save', self.engine.calls)

    def test_streamed_headroom_refusal_keeps_its_structured_code(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.svc.summary_lane.clone_enabled = True
        self.engine.atomic_main_snapshot = {"owner":"thread/turn", "tokens":65500, "hash":7, "ids":[1], "tail":4}
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({**BODY,"stream":True,"atomic_summary_clone":"thread/turn"}), {"Content-Type":"application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            wire = response.read().decode()
            self.assertIn('atomic_summary_headroom_exhausted', wire)
        finally:
            conn.close()
        self.assertNotIn('generate', self.engine.calls)
        self.assertNotIn('save', self.engine.calls)

    def test_known_restored_owner_can_report_capacity_without_claiming_clone_binding(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.svc.summary_lane.clone_enabled = True
        self.engine.atomic_main_snapshot = None
        self.engine.atomic_disk_restored_receipt = {'owner':'thread/turn','tokens':65500}
        payload = {**BODY,'atomic_summary_clone':'thread/turn'}
        status, body = self.call('/v1/chat/completions', payload)
        self.assertEqual(status, 409, body)
        self.assertEqual(body['error']['code'], 'atomic_summary_headroom_exhausted')
        status, body = self.call('/v1/chat/completions', {**payload,'atomic_summary_clone':'thread/other'})
        self.assertEqual(status, 409, body)
        self.assertEqual(body['error']['code'], 'atomic_summary_clone_stale')
        self.assertIsNone(self.engine.atomic_main_snapshot)
        self.assertNotIn('save',self.engine.calls)

    def test_bound_summary_headroom_is_refused_before_allocating_or_saving(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        self.engine.atomic_main_snapshot = {"owner":"thread/turn", "tokens":65500, "hash":7, "ids":[1], "tail":4}
        self.svc.summary_lane.clone_enabled = True
        status, body = self.call('/v1/chat/completions', {**BODY, "atomic_summary_clone":"thread/turn"})
        self.assertEqual(status, 409, body)
        self.assertEqual(body['error']['code'], 'atomic_summary_headroom_exhausted')
        self.assertNotIn('generate', self.engine.calls)
        self.assertNotIn('save', self.engine.calls)

    def test_clean_memory_refusal_keeps_bound_cache_for_retry(self):
        receipt = {"owner": "thread/turn", "tokens": 3, "hash": 7, "ids": [1,2,3], "tail": 4}
        self.engine.atomic_main_snapshot = receipt
        def state(action, *args, **kwargs):
            if action == 'caps':
                return True
            refusal = SessionRefused('memory', 'normal pressure, stable swap and RAM floor required')
            refusal.summary_refused = True
            raise refusal
        self.engine.summary_state = state
        self.svc.summary_lane.configure_state_bank(self.engine, 1000)
        self.svc.summary_lane.clone_enabled = True
        payload = {**BODY, "atomic_summary_clone":"thread/turn"}
        for _ in range(2):
            status, body = self.call('/v1/chat/completions', payload)
            self.assertEqual(status, 409, body)
            self.assertEqual(body['error']['code'], 'atomic_summary_admission_refused')
            self.assertIs(self.engine.atomic_main_snapshot, receipt)
        self.assertNotIn('generate', self.engine.calls)

    def test_native_app_cache_receipt_and_plan_refuse_foreign_origins(self):
        for origin, expected in (("tauri://localhost", 200), ("http://tauri.localhost", 200),
                                 ("https://tauri.localhost", 200), ("https://untrusted.invalid", 403)):
            for path, payload in (("history-plan", {"owner":"thread/turn", "candidates": []}),
                                  ("cache-receipt", {"owner":"thread/turn", "history_hash":"f" * 64})):
                self.engine.atomic_main_snapshot = {"owner":"thread/turn"}
                conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
                try:
                    conn.request("POST", "/v1/atomic/" + path, json.dumps(payload),
                                 {"Content-Type":"application/json", "Origin":origin})
                    response = conn.getresponse()
                    self.assertEqual(response.status, expected)
                    response.read()
                    if path == "cache-receipt":
                        self.assertEqual(self.engine.atomic_main_snapshot.get("history_hash"),
                                         "f" * 64 if expected == 200 else None)
                finally:
                    conn.close()

    def test_image_counter_matches_prepared_generation_and_reuses_embedding(self):
        from serve.test_server import ImageMarkers
        from serve.server import Vision
        class Cached(ImageMarkers.FakeVision):
            def __init__(self, directory):
                super().__init__(directory)
                self.cache = {}
                self.encodes = 0
            def encode_all(self, sources, cancel=None):
                out = []
                for source in sources:
                    key = Vision.load(source)
                    if key not in self.cache:
                        self.cache[key] = (self.rows, 7)
                        self.encodes += 1
                    out.append(self.cache[key])
                return out
            def all_cached(self, sources):
                return all(Vision.load(source) in self.cache for source in sources)
        self.svc.vision = Cached(self.tmp.name)
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "literal <|image_pad|> is text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgpkYXRh"}}]}], "think": False}
        with patch.object(self.svc, "_vision_up"):
            status, counted = self.call("/v1/atomic/prompt-token-count", payload)
            self.assertEqual(status, 200, counted)
            messages, tools, kwargs = openai_to_messages({**payload, "messages": counted["prepared_messages"]})
            ids, _, _ = self.svc.prepare(messages, tools, kwargs, 32)
            self.assertEqual(len(ids), counted["input_tokens"])
            self.assertEqual(self.svc.vision.encodes, 1)
            self.svc.embeddings.path.unlink(missing_ok=True)
        self.assertEqual(self.engine.calls, [])

    def test_foreground_count_preserves_template_tools_and_oversized_input(self):
        payload = {"messages": [{"role": "user", "content": "hello" * 14000}],
                   "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
                   "atomic_output_reserve": 1024, "think": False}
        messages, tools, kw = openai_to_messages(self.svc.with_shared(payload, "openai"))
        expected = len(self.svc.encode_prompt(messages, tools, kw, store=False))
        with patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt) as encode:
            status, counted = self.call("/v1/atomic/prompt-token-count", payload)
            self.assertEqual(status, 200, counted)
            self.assertEqual(counted["input_tokens"], expected)
            self.assertEqual(counted["context_length"], 65536)
            self.assertEqual(counted["reserved_tokens"], 1032)
            self.assertFalse(encode.call_args.kwargs["store"])
        self.assertEqual(self.engine.calls, [])

    def test_count_matches_template_without_engine_or_storing(self):
        normalized = request(BODY, count=True)
        messages, tools, kw = openai_to_messages(normalized)
        expected = len(self.svc.encode_prompt(messages, tools, kw, store=False))
        with patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt) as encode:
            status, body = self.call("/v1/atomic/summary-token-count", BODY)
            self.assertEqual(status, 200)
            self.assertEqual(body, {"input_tokens": expected, "context_length": 16384})
            self.assertFalse(encode.call_args.kwargs["store"])
        self.assertEqual(self.engine.calls, [])

    def test_oversized_count_is_allowed_but_inference_refused(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        body = {**BODY, "messages": [{"role": "user", "content": "x" * 18000}]}
        self.assertEqual(self.call("/v1/atomic/summary-token-count", body)[0], 200)
        status, error = self.call("/v1/chat/completions", body)
        self.assertEqual(status, 400)
        self.assertIn("Prompt has", error["error"]["message"])
        self.assertEqual(self.engine.calls, ["caps"])

    def test_summary_generates_between_begin_end_and_restores(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        status, body = self.call("/v1/chat/completions", BODY)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.engine.calls, ["caps", "summary_begin", "generate", "summary_end"])
        self.assertEqual(self.engine.main, "original")
        self.assertEqual(body["choices"][0]["message"]["content"], "summary")

    def test_http_foreground_preempts_summary_after_rollback(self):
        import time
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        started = threading.Event()
        original = self.engine.generate
        def generate(ids, max_new, sampling, cancel, **kw):
            if sampling.get("atomic_summary"):
                self.engine.calls.append("summary_generate")
                started.set()
                while not cancel.is_set():
                    time.sleep(0.01)
                    yield None
                return
            yield from original(ids, max_new, sampling, cancel, **kw)
        result = []
        with patch.object(self.engine, "generate", generate):
            worker = threading.Thread(target=lambda: result.append(self.call("/v1/chat/completions", BODY)))
            worker.start()
            self.assertTrue(started.wait(3))
            status, body = self.call("/v1/chat/completions", {
                "messages": BODY["messages"], "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False}})
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(status, 200, body)
        self.assertEqual(result[0][0], 409, result)
        self.assertLess(self.engine.calls.index("summary_end"), self.engine.calls.index("generate"))
        self.assertFalse(self.svc.summary_lane.poisoned)

    def test_poison_blocks_request_before_loading(self):
        self.svc.summary_lane.poisoned = True
        with patch.object(self.svc, "load", side_effect=AssertionError("must not load")):
            status, body = self.call("/v1/chat/completions", {"messages": BODY["messages"]})
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "atomic_summary_recovery_required")

    def test_invalid_summary_inputs(self):
        for update in ({"think": True}, {"max_tokens": 0}, {"max_tokens": 2049}, {"max_tokens": True},
                       {"tools": [{"type": "function"}]}, {"messages": [{"role": "tool", "content": "hi"}]},
                       {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}):
            with self.subTest(update=update):
                self.assertEqual(self.call("/v1/atomic/summary-token-count", {**BODY, **update})[0], 400)

    def test_raw_count_accepts_no_max_tokens_and_unicode(self):
        status, body = self.call("/v1/atomic/summary-token-count", {
            "atomic_summary": True, "think": False, "messages": [{"role": "user", "content": "안녕 </think>"}]})
        self.assertEqual(status, 200)
        self.assertGreater(body["input_tokens"], 0)

    def test_suspend_barrier_status_and_resume(self):
        self.svc.summary_lane.configure(self.engine, self.tmp.name, ENVELOPE)
        code, result = self.call("/v1/atomic/summary-cancel", {"suspend": True})
        self.assertEqual(code, 200, result)
        self.assertTrue(result["atomic_summary"]["suspended"])
        self.assertFalse(result["atomic_summary"]["rollback_pending"])
        self.assertEqual(self.call("/v1/chat/completions", BODY)[0], 409)
        code, status = self.call("/v1/status")
        self.assertEqual(code, 200)
        self.assertTrue(status["atomic_summary"]["suspended"])
        self.assertEqual(self.call("/v1/atomic/summary-resume", {})[0], 200)
        self.assertEqual(self.call("/v1/chat/completions", BODY)[0], 200)

    def test_owner_control_requires_private_key(self):
        for path in ("/v1/atomic/summary-cancel", "/v1/atomic/summary-resume"):
            self.assertEqual(self.call(path, {}, owner=False)[0], 403)
        self.assertFalse(self.svc.draining)
        self.assertFalse(self.svc.summary_lane.suspended)

    def test_switch_drain_is_idempotent_and_resume_reopens(self):
        code, result = self.call("/v1/atomic/summary-cancel", {"suspend": True, "drain": True})
        self.assertEqual(code, 200, result)
        self.assertTrue(result["atomic_summary"]["inference_draining"])
        self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 200)
        self.assertEqual(self.call("/v1/chat/completions", {"messages": BODY["messages"]})[0], 503)
        self.assertEqual(self.call("/v1/atomic/summary-token-count", BODY)[0], 200)
        self.assertEqual(self.call("/v1/atomic/summary-resume", {})[0], 200)
        self.assertFalse(self.svc.draining)

    def test_poisoned_idle_engine_can_arm_disposal_but_never_resume(self):
        self.svc.summary_lane.poisoned = True
        code, result = self.call("/v1/atomic/summary-cancel", {"drain": True})
        self.assertEqual(code, 200, result)
        self.assertTrue(result["atomic_summary"]["recovery_stop_ready"])
        self.assertTrue(result["atomic_summary"]["recovery_required"])
        self.assertTrue(result["atomic_summary"]["rollback_pending"])
        self.assertTrue(result["atomic_summary"]["inference_draining"])
        self.assertEqual(self.call("/v1/atomic/summary-resume", {})[0], 503)
        self.assertTrue(self.svc.draining)
        self.assertEqual(self.call("/v1/chat/completions", {"messages": BODY["messages"]})[0], 503)
        self.assertEqual(self.engine.calls, [])
        _, status = self.call("/v1/status")
        self.assertTrue(status["atomic_summary"]["recovery_stop_ready"])

    def test_poisoned_disposal_refuses_remaining_host_or_scheduler_owner(self):
        from serve.priority import WeightedScheduler, Owner
        from unittest.mock import patch
        self.svc.summary_lane.poisoned = True
        for field, value in (("summary_registered", True), ("active", threading.Event()), ("foreground", 1)):
            with self.subTest(field=field), patch.object(self.svc.summary_lane, field, value):
                self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 409)
                self.assertFalse(self.svc.draining)
        with patch.object(self.svc, "admission_reserved", 1):
            self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 409)
        scheduler = WeightedScheduler(self.svc.fifo)
        scheduler.poison("test")
        self.svc.summary_lane.scheduler = scheduler
        owner = Owner(False, threading.Event())
        for field, value in (("turn_owner", owner), ("owners", {False: owner, True: None}),
                             ("waiters", {False: [owner], True: []})):
            with self.subTest(field=field), patch.object(scheduler, field, value):
                self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 409)
                self.assertFalse(self.svc.draining)
                self.assertFalse(self.svc.summary_recovery_stop_ready)

    def test_drain_refuses_foreground_and_never_undoes_shutdown(self):
        self.assertTrue(self.svc.reserve_inference())
        try:
            self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 409)
            self.assertFalse(self.svc.draining)
        finally:
            self.svc.release_inference()
        self.assertEqual(self.call("/v1/atomic/summary-cancel", {"drain": True})[0], 200)
        self.svc.begin_drain()  # lifecycle shutdown takes ownership
        self.assertEqual(self.call("/v1/atomic/summary-resume", {})[0], 200)
        self.assertTrue(self.svc.draining)

    def test_output_reserve_error_is_exact_and_no_generation(self):
        with patch.object(self.svc, "encode_prompt", return_value=[1] * 65000):
            status, body = self.call("/v1/chat/completions", {
                "messages": BODY["messages"], "max_tokens": 20, "atomic_output_reserve": 1000})
        self.assertEqual(status, 400)
        self.assertIn("Prompt has 65000 tokens plus 1008 reserved output tokens", body["error"]["message"])
        self.assertEqual(self.engine.calls, [])

class Protocol(unittest.TestCase):
    def test_native_begin_end_progress(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SESSION 16 32\n", "SWAIT transfer 60\n", "SUMMARY_HELD 256 32 1.5\n"]
        self.assertEqual(engine.session_file("summary_begin", "/private/main", summary_envelope=ENVELOPE),
                         {"tokens": 256, "bytes": 32, "ms": 1.5})
        self.assertEqual(engine.proc.sent[-1], "SUMMARY_BEGIN 8192 67108864 134217728 /private/main\n")
        engine.script = ["SUMMARY_RELEASED 256 32 2.5\n"]
        self.assertEqual(engine.session_file("summary_end", "/private/main")["tokens"], 256)
        self.assertEqual(engine.proc.sent[-1], "SUMMARY_END /private/main\n")

    def test_missing_or_invalid_qualification_envelope_fails_closed(self):
        from serve.atomic_summary import qualification_envelope
        lane = SummaryLane()
        engine = Fake(ByteTokenizer())
        with tempfile.TemporaryDirectory() as d:
            for envelope in (None, {}, {**ENVELOPE, "max_main_tokens": 0},
                             {**ENVELOPE, "max_snapshot_bytes": True},
                             {**ENVELOPE, "max_protected_bytes": 1 << 53}):
                with self.subTest(envelope=envelope):
                    with self.assertRaises(ValueError):
                        lane.configure(engine, d, envelope)
                    self.assertFalse(lane.enabled)
                    self.assertIsNone(lane.envelope)
            with self.assertRaises(ValueError):
                qualification_envelope({"atomic_summary_transaction_qualified": True})
            self.assertEqual(qualification_envelope({"atomic_summary_qualified_" + k: v
                                                   for k, v in ENVELOPE.items()}), ENVELOPE)
        self.assertEqual(engine.calls, [])

    def test_v1_discovery_is_rejected(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SUMMARY_CAPS 1 16384\n"]
        with self.assertRaises(ValueError):
            engine.summary_control("caps")

    def test_v2_discovery_is_rejected(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SUMMARY_CAPS 2 16384\n"]
        with self.assertRaises(ValueError):
            engine.summary_control("caps")

    def test_zero_byte_empty_protocol_is_accepted(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SUMMARY_HELD 0 0 0.1\n"]
        self.assertEqual(engine.session_file("summary_begin", "/private/opaque", summary_envelope=ENVELOPE),
                         {"tokens": 0, "bytes": 0, "ms": 0.1})
        engine.script = ["SUMMARY_RELEASED 0 0 0.2\n"]
        self.assertEqual(engine.session_file("summary_end", "/private/opaque")["bytes"], 0)

    def test_empty_end_swait_fixture_uses_real_process_parser(self):
        from serve.test_slots import FakeEngine
        from serve.server import StrataEngine
        self.assertIs(FakeEngine._session_file, StrataEngine._session_file)
        engine = FakeEngine()
        fixture = ROOT / "serve/fixtures/atomic-summary/empty-end-v3.txt"
        engine.script = fixture.read_text().splitlines(keepends=True)
        self.assertEqual(engine.session_file("summary_end", "/private/opaque"),
                         {"tokens": 0, "bytes": 0, "ms": 0.2})
        self.assertFalse(engine.ended)
        self.assertFalse(engine.proc.killed)
        self.assertEqual(engine.proc.sent, ["SUMMARY_END /private/opaque\n"])

    def test_all_native_session_phase_literals_match_parser_grammar(self):
        import re
        from serve.test_slots import FakeEngine
        native = (ROOT / "src/program/generate.cpp").read_text()
        writer = (ROOT / "src/core/conversation_file.cpp").read_text()
        phases = set(re.findall(r'blocking\("([^"\n]+)"', native))
        phases.update(re.findall(r'\bphase\("([^"\n]+)"', writer))
        self.assertEqual(phases, {"emptyreset", "fingerprint", "capture", "validate", "transfer", "flush", "publish"})
        for phase in sorted(phases):
            with self.subTest(phase=phase):
                self.assertTrue(phase.isascii() and phase.isalpha())
                engine = FakeEngine()
                engine.script = [f"SWAIT {phase} 60\n", "SUMMARY_RELEASED 0 0 0.2\n"]
                self.assertEqual(engine.session_file("summary_end", "/private/opaque")["bytes"], 0)
                self.assertFalse(engine.proc.killed)

    def test_hyphenated_phase_stays_rejected_without_grammar_expansion(self):
        from serve.test_slots import FakeEngine
        from serve.server import EngineDied
        engine = FakeEngine()
        engine.script = ["SWAIT empty-reset 60\n", "SUMMARY_RELEASED 0 0 0.2\n"]
        with self.assertRaises(EngineDied):
            engine.session_file("summary_end", "/private/opaque")
        self.assertTrue(engine.proc.killed)

    def test_v2_begin_published_file_refusal_is_still_unmutated(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SERR memory 1 restore peak exceeds current RAM\n"]
        with self.assertRaises(SessionRefused) as got:
            engine.session_file("summary_begin", "/private/main", summary_envelope=ENVELOPE)
        self.assertTrue(got.exception.summary_refused)
        self.assertTrue(got.exception.published)

    def test_positive_discovery_required(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SUMMARY_CAPS 0 16384\n"]
        with self.assertRaises(ValueError):
            engine.summary_control("caps")
        engine.script = ["SUMMARY_CAPS 3 16384\n"]
        self.assertTrue(engine.summary_control("caps"))

    def test_refused_begin_proves_no_mutation(self):
        from serve.test_slots import FakeEngine
        engine = FakeEngine()
        engine.script = ["SERR memory 0 insufficient memory\n"]
        with self.assertRaises(SessionRefused) as got:
            engine.session_file("summary_begin", "/private/main", summary_envelope=ENVELOPE)
        self.assertTrue(got.exception.summary_refused)

    def test_stable_fingerprint_changes_when_artifact_changes(self):
        from serve.atomic_summary import model_fingerprint
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "engine"
            exe.write_text("version1")
            cfg = {"exe": str(exe), "args": ["--max-context", "65536"]}
            first = model_fingerprint(cfg)
            self.assertEqual(first, model_fingerprint(cfg))
            for key, value in {"atomic_summary_transaction_qualified": True,
                               "atomic_completion_review_qualified": True,
                               "atomic_summary_qualified_max_main_tokens": 8192,
                               "atomic_summary_qualified_max_snapshot_bytes": 64 << 20,
                               "atomic_summary_qualified_max_protected_bytes": 128 << 20}.items():
                with self.subTest(qualification=key):
                    self.assertNotEqual(first, model_fingerprint({**cfg, key: value}))
            exe.write_text("version2-and-change")
            self.assertNotEqual(first, model_fingerprint(cfg))
            with self.assertRaises(FileNotFoundError):
                model_fingerprint({"exe": str(exe) + "missing"})


if __name__ == "__main__":
    unittest.main()
