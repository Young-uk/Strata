"""Strict client for native resumable compute. One tagged quantum at a time."""
from __future__ import annotations

import queue
import re
import time
import uuid

from .priority import PriorityCancelled, PriorityError


class ContinuationClient:
    CAPS = ["ASYNC_CAPS", "1", "512", "1", "text"]
    IDLE_CAPS = ["ASYNC_CAPS", "2", "512", "128", "text"]
    RECOVERY_CAPS = ["ASYNC_CAPS", "3", "512", "2048", "text"]
    PREP_CAPS = ["APREP_CAPS", "1", "legacy1"]
    # This first native driver does not silently ignore unsupported GEN keys.
    SUPPORTED_KEYS = {"history", "temperature", "top_p", "top_k", "min_p", "penalty_last_n",
                      "penalty_repeat", "penalty_freq", "penalty_present", "seed"}

    def __init__(self, engine, scheduler, lane):
        self.engine, self.scheduler, self.lane = engine, scheduler, lane
        self.generation = engine.gen
        self.idle_summary_inputs = False
        self.summary_max_inputs = 1

    def poison(self, reason):
        self.scheduler.poison(reason)
        with self.lane.lock:
            self.lane.poisoned = True
            self.lane.phase = "recovery_required"
        return PriorityError("native continuation outcome uncertain; inference blocked pending restart")

    def check_generation(self):
        if self.engine.gen != self.generation or not self.engine.alive():
            raise self.poison("engine process identity changed with priority scheduling enabled")

    def _read(self, identity=None):
        """Read one meaningful line, yielding quiet heartbeats without losing owner."""
        silence = max(10.0, float(self.engine.silence_s or 300))
        deadline, beat = time.monotonic() + silence, time.monotonic()
        while True:
            self.check_generation()
            try:
                line = self.engine.lines.get(timeout=min(.5, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise self.poison("native quantum acknowledgment timed out")
                if time.monotonic() - beat >= 10:
                    beat = time.monotonic()
                    yield None
                continue
            if line is None or line.startswith(("FATAL", "ERR")):
                raise self.poison("native process ended or emitted an unscoped error")
            if line.startswith(("INFO ", "WARN ")):
                # Diagnostic traffic is not evidence of completion and cannot
                # extend an unbounded timeout on the expected acknowledgment.
                continue
            fields = line.split()
            if identity is not None and (len(fields) < 2 or fields[1] != identity):
                raise self.poison("native response belongs to another continuation")
            return fields

    @staticmethod
    def _number(text, bound=9007199254740991):
        if not re.fullmatch(r"0|[1-9][0-9]*", text) or len(text) > 16:
            raise ValueError("invalid native continuation count")
        value = int(text)
        if value > bound:
            raise ValueError("native continuation count exceeds bound")
        return value

    def _drain_owned(self, identity, admission=False):
        """Generator close during a heartbeat must not release GPU/control early."""
        while True:
            fields = self._sync_read(identity)
            if admission and len(fields) == 4 and fields[0] == "AREADY":
                return fields
            if not admission and ((len(fields) == 5 and fields[0] == "AYIELD") or
                                  (len(fields) == 4 and fields[0] == "ADONE")):
                return fields
            if not admission and ((len(fields) == 3 and fields[0] == "AT") or
                                  (len(fields) >= 3 and fields[0] == "ALP")):
                continue
            if admission and len(fields) >= 4 and fields[0] == "ASYNC_REFUSED":
                return fields
            raise self.poison("invalid response while draining an abandoned quantum")

    def _sync_read(self, identity=None):
        reader = self._read(identity)
        while True:
            try:
                next(reader)
            except StopIteration as result:
                return result.value

    def discover(self):
        with self.engine._exclusive_control():
            self.engine._send_for(self.generation, "ASYNC_CAPS")
            capabilities = self._sync_read()
            if capabilities not in (self.CAPS, self.IDLE_CAPS, self.RECOVERY_CAPS):
                raise ValueError("native weighted continuation protocol unavailable")
            self.idle_summary_inputs = capabilities != self.CAPS
            self.summary_max_inputs = int(capabilities[3])
            self.engine._send_for(self.generation, "APREP_CAPS")
            if self._sync_read() != self.PREP_CAPS:
                raise ValueError("native legacy-preparation continuation protocol unavailable")
            if self.idle_summary_inputs:
                print("[strata] summary input quanta: idle=native-max, foreground-pending=1, scheduled-foreground=512, decode=1", flush=True)

    def _sampling_keys(self, sampling):
        keys = self.engine.sampling_keys(sampling or {}).strip().split()
        # ckpt=0 is inherent for summary; foreground checkpoint/pin options must
        # be explicitly supported rather than silently altered.
        keys = [key for key in keys if key == "ckpt=0" or key.split("=", 1)[0] in self.SUPPORTED_KEYS]
        original = self.engine.sampling_keys(sampling or {}).strip().split()
        unsupported = [key.split("=", 1)[0] for key in original if key not in keys]
        if unsupported:
            raise ValueError("weighted continuation does not support native options: " + ", ".join(sorted(set(unsupported))))
        return " ".join(key for key in keys if key != "ckpt=0")

    def _read_preparation(self, identity, cancel, result, *, abandoning=False):
        """Drain legacy first-token preparation through its tagged boundary.

        STOP here means the owning user really cancelled the foreground request;
        it is never used to yield background priority.
        """
        silence = float(self.engine.silence_s or 0)
        deadline = time.monotonic() + silence if silence > 0 else None
        beat = time.monotonic()
        while True:
            self.check_generation()
            if (abandoning or cancel.is_set()) and not result.get("stop_sent"):
                try:
                    self.engine._send_for(self.generation, "STOP")
                except BaseException as exc:
                    raise self.poison("foreground preparation cancellation write uncertain") from exc
                result["stop_sent"] = True
            try:
                line = self.engine.lines.get(timeout=.5)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise self.poison("foreground preparation acknowledgment timed out")
                if time.monotonic() - beat >= 10:
                    beat = time.monotonic()
                    yield None
                continue
            if line is None or line.startswith("FATAL"):
                raise self.poison("engine ended during foreground preparation")
            if silence > 0:
                deadline = time.monotonic() + silence
            fields = line.split()
            if len(fields) == 5 and fields[0] == "APREPARED":
                if fields[1] != identity or fields[4] not in ("active", "stop", "length", "cancel", "failed"):
                    raise self.poison("invalid foreground preparation ownership")
                try:
                    result["cursor"] = self._number(fields[2], result["prompt_count"] + 1)
                    result["produced"] = self._number(fields[3], 1)
                except ValueError as exc:
                    raise self.poison(str(exc)) from exc
                result["status"] = fields[4]
                if result["produced"] != len(result["tokens"]):
                    raise self.poison("foreground first-token count mismatch")
                if fields[4] == "active" and (result["produced"] != 1 or result.get("error")):
                    raise self.poison("invalid prepared continuation")
                return
            if len(fields) == 2 and fields[0] == "T":
                if result["tokens"]:
                    raise self.poison("foreground preparation emitted multiple first tokens")
                try:
                    result["tokens"].append(self._number(fields[1], (1 << 31) - 1))
                except ValueError as exc:
                    raise self.poison(str(exc)) from exc
            elif fields[:1] == ["LP"] and result["tokens"]:
                result["logprobs"].append((result["tokens"][-1], line))
            elif fields[:1] == ["DONE"]:
                try:
                    self.engine._parse_done(line)
                    result["stats"] = dict(self.engine.last)
                except (ValueError, IndexError, TypeError) as exc:
                    raise self.poison("invalid first-token native statistics") from exc
            elif fields[:1] == ["PP"] and len(fields) >= 3:
                try:
                    cursor = self._number(fields[1], result["prompt_count"])
                    total = self._number(fields[2], result["prompt_count"])
                    if total != result["prompt_count"]:
                        raise ValueError("wrong preparation prompt length")
                    self.engine.progress = (cursor, total)
                    self.engine.progress_ms = float(fields[3]) if len(fields) > 3 else 0
                    cancel._atomic_input_progress = (cursor, total, getattr(cancel, "_atomic_input_cache", 0), int(self.engine.progress_ms))
                except (ValueError, IndexError) as exc:
                    raise self.poison("invalid preparation progress") from exc
                yield None
            # RESUME precedes prefill; REUSED follows it, including cancellation.
            # APREP uses the unchanged legacy GEN wire protocol through DONE.
            elif fields[:1] in (["RESUME"], ["REUSED"]) and len(fields) == 2:
                try:
                    self.engine.reused = self._number(fields[1], result["prompt_count"])
                    cancel._atomic_input_cache = self.engine.reused
                except ValueError as exc:
                    raise self.poison(str(exc)) from exc
            elif fields[:1] == ["ERR"]:
                result["error"] = line[4:].strip()
            elif not fields or fields[0] not in ("INFO", "WARN"):
                raise self.poison("unexpected foreground preparation response")

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        from .server import SessionRefused, LPLines, top_logprobs, EngineNonFinite
        owner = getattr(cancel, "_priority_owner", None)
        if owner is None:
            raise PriorityError("generation is missing its priority lane owner")
        if prefill_only or (owner.summary and embeddings):
            raise ValueError("weighted summary continuation requires text generation")
        if type(max_new) is not int or max_new < 1 or not ids:
            raise ValueError("invalid native continuation request")
        keys = self._sampling_keys(sampling) if owner.summary else self.engine.sampling_keys(sampling or {}).strip()
        want_lp = top_logprobs(sampling or {}) is not None
        if want_lp:
            cancel._strata_lp_lines = LPLines(cancel)
        identity = uuid.uuid4().hex
        initialized = completed = False
        in_flight = None
        produced = committed = 0
        totals = {"generated": 0, "prompt_tokens": len(ids), "prompt_ms": None,
                  "decode_ms": None, "reused": 0, "prompt_read": 0, "finish": "cancel"}
        serial = getattr(cancel, "_atomic_serial_recovery", None) if owner.summary else None
        clone = getattr(cancel, "_atomic_clone_snapshot", None) if owner.summary and not serial else None
        if serial:
            ids = [*serial["ids"], serial["tail"], *ids]
        # Previously only finished text requests were recorded. Interrupted work must
        # invalidate the former owner before it can be mistaken for this cache.
        # Completed GENI tokens and positions are copied by the native fork;
        # the summary never needs to reopen the embeddings file.
        emitted = []
        cache_owner = sampling.get("atomic_cache_owner") if not owner.summary else None
        if not owner.summary:
            self.engine.atomic_main_snapshot = None
            self.engine.atomic_disk_owner = cache_owner
        if clone:
            ids = [clone["tail"], *ids]
            totals["prompt_tokens"] = len(ids)
        clone_base = clone["tokens"] if clone else 0
        if clone: totals["cloned_tokens"] = clone_base
        self.engine.last = {}
        self.engine.progress, self.engine.progress_ms = (0, len(ids)), 0
        cancel._atomic_input_cache = 0
        cancel._atomic_input_progress = (0, len(ids), 0, 0)
        started_sent = False
        def mark_started():
            nonlocal started_sent
            if started_sent:
                return
            started_sent = True
            started = getattr(cancel, "_strata_queue_started", None)
            if started:
                started()
            cancel._strata_queue_deadline = None
        try:
            if owner.summary:
                release = yield from self.scheduler.turn(owner)
                try:
                    self.check_generation()
                    with self.engine._exclusive_control():
                        in_flight = "admission"
                        try:
                            self.engine._send_for(self.generation,
                                f"{'ARECOVER' if serial else 'AFORK' if clone else 'AGEN'} {identity} {'summary' if owner.summary and not serial else 'main'} {max_new} " +
                                (keys + " " if keys else "") + ",".join(str(int(t)) for t in ids))
                        except BaseException as exc:
                            raise self.poison("native admission write outcome is uncertain") from exc
                        fields = yield from self._read(identity)
                        in_flight = None
                        if fields[:1] == ["ASYNC_REFUSED"]:
                            if len(fields) < 4:
                                raise self.poison("malformed admission refusal")
                            refusal = SessionRefused(fields[2], " ".join(fields[3:]), False)
                            refusal.summary_refused = True
                            raise refusal
                        if len(fields) != 4 or fields[0] != "AREADY":
                            raise self.poison("invalid native admission acknowledgment")
                        try:
                            committed, prompt_count = self._number(fields[2], len(ids) + clone_base), self._number(fields[3], len(ids) + clone_base)
                            committed -= clone_base
                            prompt_count -= clone_base
                        except ValueError as exc:
                            raise self.poison(str(exc)) from exc
                        if prompt_count != len(ids):
                            raise self.poison("native prompt identity mismatch")
                        initialized = True
                        totals["reused"] = committed
                        self.engine.progress = (committed, len(ids))
                        self.engine.reused = committed
                        cancel._atomic_input_progress = (committed + clone_base, len(ids) + clone_base, committed + clone_base, 0)
                finally:
                    try:
                        if in_flight and not self.scheduler.poison_reason:
                            fields = self._drain_owned(identity, admission=True)
                            initialized = fields[0] == "AREADY"
                            in_flight = None
                    finally:
                        release()
            else:
                preparation = {"prompt_count": len(ids), "tokens": [], "logprobs": []}
                release = yield from self.scheduler.turn(owner)
                try:
                    self.check_generation()
                    with self.engine._exclusive_control():
                        in_flight = "preparation"
                        mark_started()
                        try:
                            self.engine._send_for(self.generation,
                                f"APREP {identity} {max_new} {'GENI' if embeddings else 'GEN'} " +
                                (keys + " " if keys else "") + (str(embeddings) + " " if embeddings else "") +
                                ",".join(str(int(t)) for t in ids))
                        except BaseException as exc:
                            raise self.poison("foreground preparation write outcome uncertain") from exc
                        yield from self._read_preparation(identity, cancel, preparation)
                        in_flight = None
                finally:
                    try:
                        if in_flight and not self.scheduler.poison_reason:
                            drain = self._read_preparation(identity, cancel, preparation, abandoning=True)
                            for _ in drain:
                                pass
                            in_flight = None
                        initialized = preparation.get("status") == "active"
                        completed = preparation.get("status") in ("stop", "length", "cancel", "failed")
                        produced = preparation.get("produced", 0)
                        committed = preparation.get("cursor", 0)
                        totals.update(preparation.get("stats", {}))
                        if preparation.get("status") in ("cancel", "failed"):
                            totals["finish"] = ("nonfinite" if preparation.get("stats", {}).get("finish") == "nonfinite"
                                                else "cancel" if preparation["status"] == "cancel" else "error")
                        if preparation.get("status") == "active":
                            totals["decode_ms"] = None
                            totals["finish"] = "cancel"
                            # First-token cache/IO counters are not whole-response
                            # measurements. Keep verified prompt reuse separately.
                            for key in ("hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb", "experts_streamed"):
                                totals.pop(key, None)
                    finally:
                        release()
                if preparation.get("stats", {}).get("finish") == "nonfinite":
                    raise EngineNonFinite("foreground preparation produced non-finite logits")
                if preparation.get("status") == "failed":
                    raise ValueError(preparation.get("error") or "foreground continuation preparation failed")
                if preparation.get("status") in ("stop", "length"):
                    totals["finish"] = preparation["status"]
                if preparation.get("status") == "cancel" or cancel.is_set():
                    return
                if preparation["logprobs"]:
                    if not want_lp:
                        raise self.poison("unexpected first-token logprobs")
                    for item in preparation["logprobs"]:
                        cancel._strata_lp_lines.append(item)
                for token in preparation["tokens"]:
                    emitted.append(token)
                    yield token
            mark_started()
            while not completed:
                release = yield from self.scheduler.turn(owner)
                output = []
                try:
                    self.check_generation()
                    with self.engine._exclusive_control():
                        in_flight = "quantum"
                        try:
                            # A foreground arrival after this decision waits for at most
                            # the negotiated native input block, then the ordinary weighted turn.
                            budget = (min(self.summary_max_inputs, self.scheduler.summary_input_quantum(owner, cached=True) if clone else self.scheduler.summary_input_quantum(owner))
                                      if owner.summary and self.idle_summary_inputs else None)
                            suffix = f" {budget}" if budget is not None else ""
                            self.engine._send_for(self.generation, f"ANEXT {identity}{suffix}")
                        except BaseException as exc:
                            raise self.poison("native quantum write outcome is uncertain") from exc
                        while True:
                            fields = yield from self._read(identity)
                            if len(fields) == 3 and fields[0] == "AT":
                                try:
                                    token = self._number(fields[2], (1 << 31) - 1)
                                except ValueError as exc:
                                    raise self.poison(str(exc)) from exc
                                if output:
                                    raise self.poison("native decode quantum emitted more than one token")
                                output.append(token)
                                continue
                            if len(fields) >= 3 and fields[0] == "ALP":
                                if not want_lp or len(output) != 1:
                                    raise self.poison("unexpected continuation logprobs")
                                cancel._strata_lp_lines.append((output[0], "LP " + " ".join(fields[2:]) + "\n"))
                                continue
                            if len(fields) == 5 and fields[0] == "AYIELD" and fields[2] in ("prefill", "decode"):
                                try:
                                    cursor = self._number(fields[3], len(ids) + clone_base + max_new) - clone_base
                                    count = self._number(fields[4], max_new)
                                except ValueError as exc:
                                    raise self.poison(str(exc)) from exc
                                if cursor < committed or count != produced + len(output):
                                    raise self.poison("nonmonotonic native continuation progress")
                                in_flight = None
                                committed, produced = cursor, count
                                self.engine.progress = (min(committed, len(ids)), len(ids))
                                cancel._atomic_input_progress = (min(committed, len(ids)) + clone_base, len(ids) + clone_base, totals["reused"] + clone_base, 0)
                                break
                            if len(fields) == 4 and fields[0] == "ADONE" and fields[2] in ("stop", "length"):
                                try:
                                    count = self._number(fields[3], max_new)
                                except ValueError as exc:
                                    raise self.poison(str(exc)) from exc
                                if count != produced + len(output):
                                    raise self.poison("native completion token count mismatch")
                                in_flight = None
                                produced, completed = count, True
                                totals["finish"] = fields[2]
                                break
                            raise self.poison("invalid native quantum acknowledgment")
                finally:
                    try:
                        if in_flight and not self.scheduler.poison_reason:
                            fields = self._drain_owned(identity)
                            completed = fields[0] == "ADONE"
                            try:
                                if completed:
                                    produced = self._number(fields[3], max_new)
                                    totals["finish"] = fields[2]
                                else:
                                    committed = self._number(fields[3], len(ids) + clone_base + max_new) - clone_base
                                    produced = self._number(fields[4], max_new)
                            except ValueError as exc:
                                raise self.poison(str(exc)) from exc
                            in_flight = None
                    finally:
                        release()
                # Publish progress only after acknowledgment and permit release.
                # A slow consumer cannot retain GPU ownership between quanta.
                if cancel.is_set():
                    return
                for token in output:
                    emitted.append(token)
                    yield token
                if not output and not completed:
                    yield None
        except PriorityCancelled:
            return
        finally:
            if initialized and not completed and not self.scheduler.poison_reason:
                with self.scheduler.control(owner, cleanup=True):
                    self.check_generation()
                    with self.engine._exclusive_control():
                        try:
                            self.engine._send_for(self.generation, f"ACANCEL {identity}")
                            if self._sync_read(identity) != ["ACANCELLED", identity]:
                                raise self.poison("native cancellation identity mismatch")
                        except BaseException as exc:
                            raise self.poison("native cancellation outcome is uncertain") from exc
            totals["generated"] = produced
            if owner.summary:
                totals["prompt_read"] = max(0, min(committed, len(ids)) - totals["reused"])
            if completed and totals.get("finish") in ("stop", "length"):
                totals["prompt_read"] = len(ids) - totals["reused"]
            self.engine.last = totals
            if not owner.summary and completed and totals.get("finish") == "stop" and emitted and cache_owner and not cancel.is_set():
                committed_ids = [*ids, *emitted[:-1]]
                h = 14695981039346656037
                for token in committed_ids:
                    for shift in (0, 8, 16, 24):
                        h = ((h ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
                self.engine.atomic_main_snapshot = {"owner": cache_owner, "tokens": len(committed_ids), "hash": h, "tail": emitted[-1], "ids": committed_ids}

