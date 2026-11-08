"""Opt-in clean-shutdown persistence of the drained foreground native session.

This is not an HTTP slot API: weighted continuations must have released both
owners and their summary bank before an untagged native SAVE can run.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


class DiskSession:
    def __init__(self, directory, identity, max_bytes=4 * 1024**3):
        self.directory = Path(directory)
        if not self.directory.is_absolute() or self.directory.is_symlink():
            raise ValueError('disk session directory must be an absolute non-symlink path')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.max_bytes = max_bytes
        self.manifest = self.directory / 'latest.json'

    def _record(self, require_identity=True):
        if self.manifest.is_symlink():
            raise ValueError('session manifest is a symlink')
        record = json.loads(self.manifest.read_text())
        if require_identity and record.get('identity') != self.identity:
            return None
        name = record.get('file', '')
        if not re.fullmatch(r'[a-f0-9]{32}\.session', name):
            raise ValueError('invalid session filename')
        size = record.get('bytes')
        if type(size) is not int or not 0 < size <= self.max_bytes:
            raise ValueError('invalid session size')
        path = self.directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('session file is missing or changed')
        return record

    def restore(self, engine):
        try:
            record = self._record()
        except FileNotFoundError:
            return None
        if not record:
            return None
        result = engine.session_file('restore', str(self.directory / record['file']))
        print(f"[strata] disk session restored: {result['tokens']} tokens", flush=True)
        return result

    def save(self, service):
        if not service.loaded():
            return None
        state = service.summary_lane.status()
        if any(state.get(k) for k in ('active', 'pending', 'rollback_pending', 'recovery_required', 'foreground_requests')):
            raise RuntimeError('session save skipped: summary or foreground ownership remains')
        scheduler = service.summary_lane.scheduler
        guard = scheduler.cv if scheduler is not None else contextlib.nullcontext()
        with guard:
            if scheduler is not None and (scheduler.poison_reason or scheduler.turn_owner or
                    any(scheduler.owners.values()) or any(scheduler.waiters.values())):
                raise RuntimeError('session save skipped: continuation owners remain')
            with service.fifo:
                # No new requests can enter: the caller has already drained service admission.
                return self._save(service.engine)

    def _save(self, engine):
        name = uuid.uuid4().hex + '.session'
        path = self.directory / name
        old = None
        published = False
        try:
            try:
                old = self._record(require_identity=False)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            result = engine.session_file('save', str(path))
            size = path.stat().st_size
            if result['tokens'] <= 0 or size > self.max_bytes or size + (old['bytes'] if old else 0) > self.max_bytes:
                raise ValueError('empty session or disk budget exceeded')
            os.chmod(path, 0o600)
            record = {'identity': self.identity, 'file': name, 'bytes': size, 'tokens': result['tokens']}
            temp = self.directory / (name + '.json.tmp')
            try:
                with temp.open('x', encoding='utf-8') as out:
                    os.chmod(temp, 0o600)
                    json.dump(record, out)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, self.manifest)
                published = True
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temp.unlink(missing_ok=True)
            if old and old['file'] != name:
                try:
                    (self.directory / old['file']).unlink(missing_ok=True)
                except OSError:
                    pass
            print(f"[strata] disk session saved: {result['tokens']} tokens, {size} bytes", flush=True)
            return result
        except BaseException:
            # Never invalidate the previous published checkpoint on a failed save.
            if not published:
                path.unlink(missing_ok=True)
            raise
