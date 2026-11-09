"""Opt-in serialized summary transactions: disk snapshots or separate state banks.

The default is disabled. Native protocol discovery and explicit local qualification
are both required. A failed rollback poisons admission rather than guessing that a
subsequent GEN can repair a partially restored conversation.
"""
from __future__ import annotations

import contextlib
import gc
import functools
import re
import hashlib
import json
import os
import tempfile
import stat
import threading
import uuid
from pathlib import Path
from serve.priority import PriorityCancelled

CONTEXT = 16384
OUTPUT = 2048


class SummaryError(RuntimeError):
    def __init__(self, message, status=503, code="atomic_summary_unavailable"):
        super().__init__(message)
        self.status, self.code = status, code


def request(req, *, count=False):
    """Validate the same wire request for counting and generation, without mutation."""
    if req.get("atomic_summary") is not True or req.get("think") is not False:
        raise ValueError("summary requests require atomic_summary:true and think:false")
    if "atomic_summary_clone" in req and (not isinstance(req["atomic_summary_clone"], str) or
        re.fullmatch(r"[A-Za-z0-9_:/.-]{1,256}", req["atomic_summary_clone"]) is None):
        raise ValueError("invalid cached summary owner")
    if "atomic_background_task" in req and type(req["atomic_background_task"]) is not bool:
        raise ValueError("atomic_background_task must be a boolean")
    if "atomic_summary_recovery" in req and type(req["atomic_summary_recovery"]) is not bool:
        raise ValueError("atomic_summary_recovery must be a boolean")
    if req.get("tools") or req.get("strata_mcp") or req.get("response_format"):
        raise ValueError("summary requests are text-only, without tools or response_format")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("summary messages must be a nonempty list")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise ValueError("summary messages must use system, user or assistant roles")
        if m.get("tool_calls") or m.get("reasoning_content"):
            raise ValueError("summary messages cannot contain tool calls or reasoning")
        content = m.get("content")
        if isinstance(content, list):
            if not all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
                       for p in content):
                raise ValueError("summary messages must contain only text")
        elif not isinstance(content, str):
            raise ValueError("summary messages must contain only text")
    cap = req.get("max_tokens", OUTPUT)
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= OUTPUT:
        raise ValueError(f"summary max_tokens must be between 1 and {OUTPUT}")
    if "stream" in req and type(req["stream"]) is not bool:
        raise ValueError("summary stream must be a boolean")
    # Do not inherit foreground thinking, MCP, or cache hints. Exact same kwargs
    # are used for count and inference, regardless of shared server defaults.
    out = dict(req)
    out.update(max_tokens=cap, strata_checkpoint=False, reasoning_effort="none",
               chat_template_kwargs={"enable_thinking": False})
    out.pop("strata_prefix", None)
    out.pop("max_completion_tokens", None)
    return out


def model_fingerprint(cfg):
    """Stable local artifact change detector, not a cryptographic model-content claim.

    Device numbers are mount-session identities and change across reboots.
    Keep them in transient hash-race checks, never in the persisted model pin.
    Paths stay private inside the digest. Inode, ctime, file size and nanosecond mtime plus the
    resolved generation config distinguish replacements without reading 40GB.
    Tokenizer/template bytes are small enough to hash completely.
    """
    cwd = Path(cfg.get("cwd") or ".").resolve()
    args = list(cfg.get("args") or [])
    paths = []
    for key in ("exe", "tokenizer"):
        value = cfg.get(key)
        if value:
            paths.append(value)
    for i, arg in enumerate(args[:-1]):
        if arg in ("--native", "--ple-gguf", "--pack", "--expert-profile", "--mtp"):
            paths.append(args[i + 1])
    vision = cfg.get("vision") or {}
    if isinstance(vision, dict):
        paths.extend(vision[key] for key in ("exe", "mmproj", "model") if vision.get(key))
    artifacts = []
    for value in paths:
        path = Path(value).expanduser()
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        stat = path.stat()  # fail closed: an unverified path is not a model identity
        artifacts.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        if path.is_dir():
            # Bind every pack member, not only directory mtime/tokenizer names.
            for member in sorted(path.rglob('*')):
                if member.is_file():
                    info = member.stat()
                    artifacts.append((str(member.resolve()), info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
            for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"):
                child = path / name
                if child.is_file():
                    artifacts.append((str(child), hashlib.sha256(child.read_bytes()).hexdigest()))
    if not artifacts:
        raise ValueError("model fingerprint needs verified local artifacts")
    payload = {"contract": "serial-snapshot-v3",
               "qualification": {key: cfg.get(key) for key in (
                   "atomic_summary_transaction_qualified", "atomic_completion_review_qualified",
                   "atomic_summary_qualified_max_main_tokens", "atomic_summary_qualified_max_snapshot_bytes",
                   "atomic_summary_qualified_max_protected_bytes")}, "artifacts": artifacts, "args": args, "user_args": cfg.get("user_args"),
               "backend": cfg.get("backend"), "arch": cfg.get("arch", "qwen4exp"),
               "env": cfg.get("env", {}),
               "inherited_engine_env": {key: value for key, value in os.environ.items()
                   if not cfg.get("atomic_observed_spawn") and key.startswith("STRATA_") and not key.startswith("STRATA_OWNER_") and key not in cfg.get("env", {})},
               "sampling": cfg.get("sampling", {}),
               "vision": cfg.get("vision"), "reasoning_effort": cfg.get("reasoning_effort"), "effort_position": cfg.get("effort_position")}
    if cfg.get("atomic_summary_mode") == "state-bank-v1":
        payload["contract"] = "shared-weights-state-bank-v4"
        payload["state_bank"] = {key: cfg.get(key) for key in (
            "atomic_summary_mode", "atomic_summary_clone_enabled", "atomic_summary_state_bank_enabled",
            "atomic_summary_state_bank_max_bytes", "atomic_summary_state_bank_provenance",
            "atomic_summary_scheduler", "atomic_summary_foreground_weight")}
    return "strata-stat-v2:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


QUALIFICATION_PIN_KEYS = (
    "atomic_summary_expected_native_sha256",
    "atomic_summary_expected_model_fingerprint",
    "atomic_summary_expected_backend_sha256",
)


def _file_identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@functools.lru_cache(maxsize=32)
def _stable_file_sha(path, identity):
    path = Path(path)
    if _file_identity(path) != identity:
        raise ValueError("qualification file changed before hashing")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if _file_identity(path) != identity:
        raise ValueError("qualification file changed while hashing")
    return digest.hexdigest()



def runtime_qualification_config(cfg, engine):
    """Observe actual post-fallback/retry spawn, never intended raw config alone."""
    spawn = getattr(engine, "spawn", None)
    if not isinstance(spawn, tuple) or len(spawn) < 5:
        raise ValueError("actual_engine_spawn_unavailable")
    exe, args, cwd, _, env = spawn[:5]
    if not isinstance(exe, str) or not isinstance(args, list) or not all(isinstance(v, str) for v in args) or not isinstance(env, dict):
        raise ValueError("actual_engine_spawn_invalid")
    # Bind actual arithmetic/device/loader environment and all explicit config
    # overrides. Exclude ownership secrets and unrelated GUI/session variables.
    prefixes = ("STRATA_", "GGML_", "CUDA_", "HIP_", "HSA_", "ROCR_", "OMP_", "MKL_", "METAL_", "MTL_", "DYLD_", "LD_")
    explicit = set((cfg.get("env") or {}).keys())
    relevant_env = {key: str(value) for key, value in env.items()
                    if (key.startswith(prefixes) or key in explicit or key == "PATH")
                    and not key.startswith("STRATA_OWNER_")}
    return {**cfg, "exe": exe, "args": list(args), "cwd": cwd or cfg.get("cwd"),
            "env": relevant_env, "atomic_observed_spawn": True}


def qualified_runtime_pins(cfg):
    """Compute reviewed pins without mutating configuration or launching engines.

    Call on the FINAL resolved configuration. Pin keys are intentionally absent
    from model_fingerprint's payload, avoiding a self-reference. Model hashes are
    qualification-time evidence; live metadata detects ordinary artifact drift.
    """
    cwd = Path(cfg.get("cwd") or ".").resolve()
    exe = Path(cfg["exe"]).expanduser()
    exe = ((cwd / exe) if not exe.is_absolute() else exe).resolve()
    here = Path(__file__).resolve().parent
    source_hashes = {name: _stable_file_sha(str(here / name), _file_identity(here / name))
                     for name in ("server.py", "atomic_summary.py", "priority.py", "priority_client.py", "disk_session.py", "history_plan.py")}
    return dict(zip(QUALIFICATION_PIN_KEYS, (
        _stable_file_sha(str(exe), _file_identity(exe)),
        model_fingerprint(cfg),
        hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
    )))


def qualification_mismatch(cfg):
    """Return a safe diagnostic, never turn optional drift into a foreground error."""
    for key in QUALIFICATION_PIN_KEYS:
        value = cfg.get(key)
        pattern = r"strata-stat-v2:[0-9a-f]{64}" if key.endswith("model_fingerprint") else r"[0-9a-f]{64}"
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            return "missing_or_invalid_" + key
    try:
        observed = qualified_runtime_pins(cfg)
    except (OSError, ValueError, TypeError, KeyError):
        return "qualification_identity_unavailable"
    return next(("changed_" + key for key in QUALIFICATION_PIN_KEYS if cfg[key] != observed[key]), None)


def configure_optional_lane(lane, engine, cfg, *, engine_name, arch, groups=False):
    """Optional feature only. Invalid qualification never blocks normal startup."""
    lane.enabled = False
    lane.disabled_reason = None
    bank = cfg.get("atomic_summary_mode") == "state-bank-v1"
    if not bank and cfg.get("atomic_summary_mode") not in (None, "serial-snapshot-v1"):
        lane.disabled_reason = "unsupported_summary_mode"
        return False
    if (cfg.get("atomic_summary_state_bank_enabled") if bank else cfg.get("atomic_summary_transaction_qualified")) is not True:
        return False
    try:
        reason = qualification_mismatch(runtime_qualification_config(cfg, engine))
        if reason:
            raise ValueError(reason)
        if engine_name != "strata" or arch != "qwen4exp" or groups:
            raise ValueError("unsupported_summary_engine_configuration")
        if bank:
            if cfg.get("atomic_summary_transaction_qualified") is not False or cfg.get("atomic_summary_state_bank_provenance") != "experimental-user-validation":
                raise ValueError("unsupported_summary_state_bank_provenance")
            lane.configure_state_bank(engine, cfg.get("atomic_summary_state_bank_max_bytes"),
                binding_check=lambda: qualification_mismatch(runtime_qualification_config(cfg, engine)))
            lane.disk_recovery_enabled = cfg.get("atomic_summary_disk_recovery_enabled") is True
            return True
        directory = cfg.get("atomic_summary_snapshot_path")
        if not isinstance(directory, str) or not os.path.isabs(directory):
            raise ValueError("invalid_summary_snapshot_directory")
        lane.configure(engine, directory, qualification_envelope(cfg),
                       binding_check=lambda: qualification_mismatch(runtime_qualification_config(cfg, engine)))
        return True
    except Exception as exc:
        lane.enabled = False
        lane.disabled_reason = "qualification_disabled: " + type(exc).__name__
        # Keep detailed local reason without exposing paths or file contents.
        if isinstance(exc, ValueError) and str(exc).startswith(("missing_or_invalid_", "changed_", "qualification_identity_", "unsupported_summary_", "invalid_summary_")):
            lane.disabled_reason = str(exc)
        return False


ENVELOPE_KEYS = ("max_main_tokens", "max_snapshot_bytes", "max_protected_bytes")
SCRATCH_RESERVE_BYTES = 1 << 30


def validate_envelope(envelope):
    if not isinstance(envelope, dict) or set(envelope) != set(ENVELOPE_KEYS):
        raise ValueError("summary qualification requires explicit main-token, snapshot-byte and protected-byte bounds")
    for key in ENVELOPE_KEYS:
        value = envelope[key]
        # Exact JSON integers, with a signed native prompt-position bound.
        limit = (1 << 31) - 1 if key == "max_main_tokens" else (1 << 53) - 1
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"summary qualified {key} must be a positive bounded integer")
    return dict(envelope)


def qualification_envelope(cfg):
    return validate_envelope({key: cfg.get("atomic_summary_qualified_" + key) for key in ENVELOPE_KEYS})


def snapshot_mode(result, path):
    """CAPS3 proves the meaning of zero/zero, not mere absence of a file."""
    if not isinstance(result, dict):
        raise ValueError("missing native transaction identity")
    counts = tuple(result.get(key) for key in ("tokens", "bytes"))
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
        raise ValueError("invalid native transaction identity")
    if counts == (0, 0):
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return "empty", counts
        if not stat.S_ISREG(info.st_mode) or info.st_size != 0:
            raise ValueError("EMPTY transaction has an unexpected snapshot artifact")
        return "empty", counts
    if min(counts) <= 0:
        raise ValueError("mixed EMPTY/DISK transaction identity")
    info = os.lstat(path)  # known DISK must exist BEFORE any summary generation
    if not stat.S_ISREG(info.st_mode) or info.st_size != counts[1]:
        raise ValueError("main-session snapshot missing or inconsistent")
    return "disk", counts


class SummaryLane:
    def __init__(self):
        self.enabled = False
        self.clone_enabled = False
        self.scheduler = None
        self.mode = "serial-snapshot-v1"
        self.max_slot_bytes = None
        self.directory = None
        self.envelope = None
        self.binding_check = None
        self.disabled_reason = None
        self.lock = threading.Lock()
        self.foreground = 0
        self.active = None
        self.summary_registered = False
        self.pending_cancel = None
        self.suspended = False
        self.phase = "idle"
        self.poisoned = False
        self.recovery_path = None
        self.transaction_mode = None

    def configure(self, engine, directory, envelope=None, binding_check=None):
        """Called only for explicit qualified config, before serving clients."""
        self.enabled = False  # failed discovery must never retain a stale capability
        self.envelope = None
        self.binding_check = binding_check
        self.disabled_reason = None
        envelope = validate_envelope(envelope)
        if envelope["max_main_tokens"] > getattr(engine, "max_context", 0):
            raise ValueError("qualified main-token bound exceeds engine context")
        if getattr(engine, "max_context", 0) < CONTEXT:
            raise ValueError("summary transactions require at least 16K engine context")
        if getattr(engine, "batch", 0) or getattr(engine, "session_save_reclaim", False):
            raise ValueError("summary transactions require single sequence and no session-save-reclaim")
        path = Path(directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        engine.summary_control("caps")
        self.mode, self.max_slot_bytes = "serial-snapshot-v1", None
        self.directory, self.envelope, self.enabled = str(path), envelope, True

    def configure_state_bank(self, engine, max_bytes, binding_check=None):
        """Experimental separate mutable state, shared weights, serialized compute."""
        self.enabled = False
        self.envelope = None
        self.directory = None
        if type(max_bytes) is not int or not 0 < max_bytes <= 9007199254740991:
            raise ValueError("invalid_summary_state_bank_budget")
        if getattr(engine, "max_context", 0) < CONTEXT or getattr(engine, "batch", 0) or getattr(engine, "session_save_reclaim", False):
            raise ValueError("unsupported_summary_state_bank_configuration")
        engine.summary_state("caps")
        self.mode = "state-bank-v1"
        self.max_slot_bytes = max_bytes
        self.binding_check = binding_check
        self.disabled_reason = None
        self.enabled = True

    def status(self):
        with self.lock:
            recovery = self.poisoned or bool(self.scheduler and self.scheduler.poison_reason)
            return {"enabled": self.enabled, "disabled_reason": self.disabled_reason, "pending": self.summary_registered,
                    "qualified_envelope": dict(self.envelope) if self.enabled and self.envelope else None,
                    "active": self.active is not None, "phase": self.phase,
                    "transaction_mode": self.transaction_mode,
                    "rollback_pending": self.active is not None or recovery,
                    "recovery_required": recovery, "suspended": self.suspended,
                    "foreground_requests": self.foreground}

    def cancel_summary(self, suspend=False):
        with self.lock:
            if suspend:
                self.suspended = True
            if self.pending_cancel is not None:
                self.pending_cancel.set()
            if self.active is not None:
                self.active.set()
        return self.status()

    def resume_summaries(self):
        self.check_healthy()
        with self.lock:
            self.suspended = False
        return self.status()

    def foreground_arrived(self):
        with self.lock:
            if self.active is not None and self.scheduler is None:
                self.active.set()

    def check_healthy(self):
        with self.lock:
            if self.poisoned or (self.scheduler is not None and self.scheduler.poison_reason):
                raise SummaryError("engine state is uncertain; restart the engine before retrying",
                                   code="atomic_summary_recovery_required")

    @contextlib.contextmanager
    def register(self, summary, cancel):
        with self.lock:
            if self.poisoned:
                raise SummaryError("engine state is uncertain; restart the engine before retrying",
                                   code="atomic_summary_recovery_required")
            if summary:
                if not self.enabled:
                    raise SummaryError("isolated summary transactions have not been qualified on this server", 501)
                if self.suspended:
                    raise SummaryError("summary scheduling is suspended", 409, "atomic_summary_suspended")
                if self.summary_registered:
                    raise SummaryError("a summary request is already pending", 409, "atomic_summary_busy")
                self.summary_registered = True
                self.pending_cancel = cancel
            else:
                self.foreground += 1
                if self.active is not None and self.scheduler is None:
                    self.active.set()
        try:
            yield
        finally:
            with self.lock:
                if summary:
                    self.summary_registered = False
                    self.pending_cancel = None
                else:
                    self.foreground -= 1

    @contextlib.contextmanager
    def serial_recovery(self, engine, cancel):
        """Emergency single-state work, sealed by a verified disk rollback.

        Only a bound complete token receipt qualifies. Never replay partial or
        guessed history, and never overlap another foreground continuation.
        """
        scheduler = self.scheduler
        owner = getattr(cancel, "_priority_owner", None)
        snapshot = getattr(cancel, "_atomic_clone_snapshot", None)
        if not scheduler or not owner or not owner.recovery or not snapshot or                 not isinstance(snapshot.get("ids"), list) or len(snapshot["ids"]) != snapshot.get("tokens"):
            raise SummaryError("A complete current conversation cache is required for disk recovery; originals retained", 409,
                               "atomic_summary_admission_refused")
        path = None
        saved = None
        held = False
        restored = False
        try:
            with scheduler.control(owner):
                with scheduler.cv:
                    foreground = scheduler.owners[False]
                    if foreground is not None and not foreground.closed:
                        raise SummaryError("Recovery is waiting for the active conversation to finish; originals retained", 409,
                                           "atomic_summary_admission_refused")
                    scheduler.exclusive_owner = owner
                    held = True
                current = getattr(engine, "atomic_main_snapshot", None)
                if current is not None and any(current.get(key) != snapshot.get(key) for key in ("owner", "tokens", "hash")):
                    raise SummaryError("Conversation cache changed before disk recovery; originals retained", 409,
                                       "atomic_summary_clone_stale")
                fd, path = tempfile.mkstemp(prefix="atomic-recovery-", suffix=".session")
                os.close(fd)
                self.phase = "saving"
                self.transaction_mode = "serial-disk-recovery"
                saved = engine.session_file("save", path)
                if saved.get("tokens") != snapshot["tokens"] or saved.get("bytes", 0) <= 0:
                    raise ValueError("recovery snapshot does not match the bound main cache")
                cancel._atomic_serial_recovery = snapshot
                self.transaction_mode = "serial-disk-recovery"
                self.phase = "generating"
            try:
                yield
            finally:
                self.phase = "restoring"
                with scheduler.control(owner, cleanup=True):
                    ended = engine.session_file("restore", path)
                    if (ended.get("tokens"), ended.get("bytes")) != (saved["tokens"], saved["bytes"]):
                        raise ValueError("recovery rollback identity mismatch")
                    restored = True
        except BaseException as exc:
            # A failed SAVE never selected a different main. Once SAVE succeeded,
            # any uncertain outcome keeps the artifact and blocks further work.
            clean_refusal = saved is None and getattr(exc, "published", None) is False
            if (saved is not None and not restored) or (saved is None and path and not clean_refusal):
                self.poisoned, self.recovery_path = True, path
                scheduler.poison("serial recovery rollback is uncertain")
                raise SummaryError("Disk recovery could not restore the original state; snapshot retained and inference blocked",
                                   code="atomic_summary_recovery_required") from exc
            raise
        finally:
            if hasattr(cancel, "_atomic_serial_recovery"):
                del cancel._atomic_serial_recovery
            if held and not self.poisoned:
                with scheduler.cv:
                    scheduler.exclusive_owner = None
                    scheduler.cv.notify_all()
            if path and not self.poisoned:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    @contextlib.contextmanager
    def transaction(self, engine, cancel):
        """Caller owns the engine FIFO. Rollback completes before FIFO release."""
        if self.binding_check is not None:
            try:
                reason = self.binding_check()
            except Exception:
                reason = "qualification_identity_unavailable"
            if reason:
                with self.lock:
                    self.enabled = False
                    self.disabled_reason = reason
                raise SummaryError("summary qualification changed; foreground remains available", 409,
                                   "atomic_summary_admission_refused")
        with self.lock:
            if self.poisoned:
                raise SummaryError("main session recovery is required", code="atomic_summary_recovery_required")
            if (self.foreground and self.scheduler is None) or cancel.is_set():
                raise SummaryError("foreground work takes priority; retry the summary when idle", 409,
                                   "atomic_summary_interrupted")
            self.active = cancel
            self.phase = "switching" if self.mode == "state-bank-v1" else "saving"
            self.transaction_mode = "unknown"
        @contextlib.contextmanager
        def native_turn(cleanup=False):
            if self.scheduler is None:
                yield
            else:
                owner = getattr(cancel, "_priority_owner", None)
                if owner is None:
                    raise SummaryError("summary is missing its priority owner")
                with self.scheduler.control(owner, cleanup=cleanup):
                    try:
                        yield
                    except BaseException as exc:
                        if not getattr(exc, "summary_refused", False):
                            # Stop already-admitted foreground turns before the
                            # compute permit is released after an uncertain swap.
                            self.scheduler.poison("uncertain summary bank transition")
                            with self.lock:
                                self.poisoned = True
                        raise

        # Unique private file. Never reuse a caller-controlled filename or save
        # reclaim mode: it would evict the very checkpoints being protected.
        path = None
        bank = self.mode == "state-bank-v1"
        txid = uuid.uuid4().hex if bank else None
        try:
            if not bank:
                fd, path = tempfile.mkstemp(prefix="atomic-main-", suffix=".session", dir=self.directory)
                os.close(fd)
            try:
                if bank:
                    for attempt in range(2):
                        try:
                            with native_turn():
                                snapshot = getattr(cancel, "_atomic_clone_snapshot", None)
                                begun = engine.summary_state("fork", txid, max_slot_bytes=self.max_slot_bytes, binding=snapshot) if snapshot else engine.summary_state("begin", txid, max_slot_bytes=self.max_slot_bytes)
                            break
                        except BaseException as exc:
                            refused = getattr(exc, "summary_refused", False)
                            reason = str(exc)[:240] if refused else "uncertain native transition"
                            print(f"[atomic-summary] bank admission attempt {attempt + 1}: {reason}", flush=True)
                            # A clean refusal after shared preparation never selected a
                            # summary bank. Re-enter the full native safety gate once,
                            # with already prepared resources and a fresh stable-swap
                            # observation. Never retry arbitrary pressure/OOM refusals
                            # or an uncertain transition, and never evaluate a token.
                            if (attempt == 0 and refused and not cancel.is_set() and
                                    reason == "pressure or memory changed during shared resource preparation"):
                                continue
                            raise
                    mode, counts = "state-bank", begun
                else:
                    begun = engine.session_file("summary_begin", path, summary_envelope=self.envelope)
                    mode, counts = snapshot_mode(begun, path)
                with self.lock:
                    self.transaction_mode = mode
            except PriorityCancelled as exc:
                # This exception comes from waiting for the compute permit;
                # native BEGIN has not been sent, so foreground stays healthy.
                raise SummaryError("summary cancelled before bank admission", 409,
                                   "atomic_summary_interrupted") from exc
            except BaseException as exc:
                # A structured native refusal proves BEGIN did not mutate state.
                if not getattr(exc, "summary_refused", False):
                    with self.lock:
                        self.poisoned, self.recovery_path = True, path
                refusal_messages = {
                    "pressure or memory changed during shared resource preparation":
                        "Summary resources are ready, but memory pressure or swap activity is not stable; originals retained",
                    "summary bank allocation refused":
                        "Not enough memory for the separate summary state; originals retained",
                }
                memory_refusals = {
                    "normal pressure, stable swap and RAM floor required",
                    "pressure or available memory changed during bank allocation",
                    "pressure or memory changed during quiescence",
                    "pressure or memory refused shared resource preparation",
                }
                reason = str(exc)
                if reason in memory_refusals or reason in refusal_messages:
                    # Collect only unreachable Python request objects. Live
                    # conversation/vision/cache owners are never discarded.
                    gc.collect()
                owner = getattr(cancel, "_priority_owner", None)
                if (getattr(exc, "summary_refused", False) and owner is not None and owner.recovery and getattr(self, "disk_recovery_enabled", False) and
                        (reason in memory_refusals or reason in refusal_messages)):
                    with self.serial_recovery(engine, cancel):
                        yield
                    return
                message = refusal_messages.get(reason, "Summary is waiting for normal memory pressure, stable swap and available RAM; originals retained"
                    if reason in memory_refusals else "summary state could not be established; main conversation preserved")
                failure = SummaryError(message
                                   if getattr(exc, "summary_refused", False) else
                                   "summary begin outcome is uncertain; inference blocked pending recovery",
                                   409 if getattr(exc, "summary_refused", False) else 503,
                                   "atomic_summary_admission_refused" if getattr(exc, "summary_refused", False) else
                                   "atomic_summary_recovery_required")
                failure.preserve_clone = bool(getattr(exc, "summary_refused", False) and
                                              (reason in memory_refusals or reason in refusal_messages))
                raise failure from exc
            with self.lock:
                self.phase = "generating"
            try:
                if cancel.is_set():
                    raise SummaryError("summary interrupted before generation", 409, "atomic_summary_interrupted")
                yield
            finally:
                with self.lock:
                    self.phase = "restoring"
                try:
                    with native_turn(cleanup=True):
                        ended = engine.summary_state("end", txid) if bank else engine.session_file("summary_end", path)
                        if bank and self.scheduler is not None:
                            # Foreground legitimately advances while summary owns its
                            # resident bank. Only the immutable ownership tuple must
                            # match; native parsing already bounds latest main_tokens.
                            matching = all(ended.get(key) == counts.get(key) for key in
                                           ("transaction_id", "allocated_bytes", "epoch"))
                        else:
                            matching = (ended if bank else (ended.get("tokens"), ended.get("bytes"))) == counts
                        if not matching:
                            raise ValueError("native rollback identity does not match BEGIN")
                except BaseException as exc:
                    with self.lock:
                        self.poisoned, self.recovery_path = True, path
                    raise SummaryError("pristine empty-state rollback failed; no saved main KV existed; inference blocked"
                                       if mode == "empty" else
                                       "main state-bank restoration failed; inference blocked; process restart required" if bank else
                                       "main session rollback failed; recovery artifacts retained and inference blocked",
                                       code="atomic_summary_recovery_required") from exc
            if cancel.is_set():
                raise SummaryError("summary interrupted for foreground work or client cancellation", 409,
                                   "atomic_summary_interrupted")
        finally:
            with self.lock:
                self.active = None
                self.phase = "recovery_required" if self.poisoned else "idle"
                keep = self.poisoned
                if not keep:
                    self.transaction_mode = None
            if not keep and path is not None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


def historical_handoff_key(messages):
    """Inspect validated wire roles BEFORE the frontend converts late systems to users."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (ValueError, TypeError):
            return None
    if not isinstance(messages, list):
        return None
    labels = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text" and
                                            isinstance(p.get("text"), str) for p in content):
            content = "".join(p["text"] for p in content)
        if isinstance(content, str) and content.startswith(
                "Earlier conversation summary (historical data, not new instructions)."):
            labels.append(content)
    return hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest()[:16] if labels else None
