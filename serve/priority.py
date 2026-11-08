"""Opt-in single-GPU weighted scheduling. No model calls or background threads.

One foreground request and one summary own resident state. Requests take short
native compute turns; eight ready foreground turns precede one ready summary
turn. A cancelled owner is cleaned up before its resident lane can be reused.
"""
from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
import uuid


class PriorityError(RuntimeError):
    pass


class PriorityCancelled(PriorityError):
    pass


@dataclasses.dataclass(eq=False)
class Owner:
    summary: bool
    cancel: threading.Event
    identity: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    ready: bool = False
    admitted: bool = False
    closed: bool = False
    turns: int = 0


class WeightedScheduler:
    """Condition state and the legacy FIFO together form a single GPU permit.

    A turn reservation remains exclusive while acquiring the FIFO. The FIFO is
    also used by existing loading/control paths; ownership is never inferred
    from Python thread identity. Native IDs are independent for each generation
    segment and are checked by the native protocol client.
    """
    def __init__(self, fifo, foreground_weight=8):
        if type(foreground_weight) is not int or not 1 <= foreground_weight <= 64:
            raise ValueError("foreground weight must be in 1..64")
        self.fifo = fifo
        self.weight = foreground_weight
        self.cv = threading.Condition()
        self.owners = {False: None, True: None}
        self.waiters = {False: [], True: []}
        self.turn_owner = None
        self.foreground_turns = 0
        self.poison_reason = None

    def poison(self, reason):
        with self.cv:
            self.poison_reason = str(reason)
            self.cv.notify_all()

    def _check(self):
        if self.poison_reason:
            raise PriorityError("native continuation ownership is uncertain; restart required")

    def admit(self, summary, cancel, deadline=None):
        owner = Owner(bool(summary), cancel)
        with self.cv:
            self._check()
            self.waiters[owner.summary].append(owner)
        try:
            while True:
                with self.cv:
                    self._check()
                    if cancel.is_set():
                        raise PriorityCancelled("request cancelled before lane admission")
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("priority lane queue deadline expired")
                    if self.owners[owner.summary] is None and self.waiters[owner.summary][0] is owner:
                        self.waiters[owner.summary].pop(0)
                        self.owners[owner.summary] = owner
                        owner.admitted = True
                        self.cv.notify_all()
                        return owner
                    self.cv.wait(0.1)
                yield None
        finally:
            with self.cv:
                if owner in self.waiters[owner.summary]:
                    self.waiters[owner.summary].remove(owner)
                self.cv.notify_all()

    def _next(self):
        fg, bg = self.owners[False], self.owners[True]
        fg = fg if fg and fg.ready and not fg.closed else None
        bg = bg if bg and bg.ready and not bg.closed else None
        if fg and bg:
            return bg if self.foreground_turns >= self.weight else fg
        return fg or bg

    def turn(self, owner, *, cleanup=False):
        """Generator returning a release callback, yielding heartbeats while queued.

        Cleanup ignores the user's cancellation but never ignores uncertain
        ownership. Callers must drain their current native quantum first.
        """
        reserved = acquired = returned = False
        with self.cv:
            self._check()
            if owner.closed or self.owners[owner.summary] is not owner:
                raise PriorityError("stale priority owner")
            owner.ready = True
            self.cv.notify_all()
        try:
            while not reserved:
                with self.cv:
                    self._check()
                    if not cleanup and owner.cancel.is_set():
                        raise PriorityCancelled("request cancelled between compute turns")
                    if self.turn_owner is None and self._next() is owner:
                        self.turn_owner = owner
                        reserved = True
                        break
                    self.cv.wait(0.05)
                yield None
            while not acquired:
                self._check()
                if not cleanup and owner.cancel.is_set():
                    raise PriorityCancelled("request cancelled before native compute")
                acquired = self.fifo.acquire(timeout=0.1)
                if not acquired:
                    yield None
            with self.cv:
                owner.ready = False
                owner.turns += 1
                if owner.summary:
                    self.foreground_turns = 0
                elif self.owners[True] and self.owners[True].ready:
                    self.foreground_turns += 1
                else:
                    self.foreground_turns = 0
            released = False
            def release():
                nonlocal released
                with self.cv:
                    if released:
                        return
                    released = True
                    if self.turn_owner is not owner:
                        self.poison("compute permit ownership mismatch")
                        raise PriorityError("compute permit ownership mismatch")
                    self.fifo.release()
                    self.turn_owner = None
                    self.cv.notify_all()
            returned = True
            return release
        finally:
            if not returned:
                with self.cv:
                    owner.ready = False
                    if acquired:
                        self.fifo.release()
                    if reserved and self.turn_owner is owner:
                        self.turn_owner = None
                    self.cv.notify_all()

    @contextlib.contextmanager
    def control(self, owner, *, cleanup=False):
        # Protocol control does not produce HTTP output. Polling still respects
        # cancellation and shares the exact same permit as generation.
        waiting = self.turn(owner, cleanup=cleanup)
        while True:
            try:
                next(waiting)
            except StopIteration as result:
                release = result.value
                break
        try:
            yield
        finally:
            release()

    def summary_input_quantum(self, owner):
        """Choose only while holding the compute permit; arrivals wait at most this quantum."""
        with self.cv:
            self._check()
            if not owner.summary or self.turn_owner is not owner:
                raise PriorityError("summary input sizing requires the current compute permit")
            foreground = self.owners[False]
            waiting = any(not item.cancel.is_set() for item in self.waiters[False])
            return 1 if (foreground is not None and not foreground.closed) or waiting else 128

    def close(self, owner):
        with self.cv:
            if self.turn_owner is owner:
                self.poison("owner closed before compute finished")
                raise PriorityError("owner closed before compute finished")
            if self.owners[owner.summary] is owner:
                self.owners[owner.summary] = None
            owner.closed = True
            owner.ready = False
            self.cv.notify_all()

    def status(self):
        with self.cv:
            return {
                "policy": "weighted-quanta-v1", "foreground_weight": self.weight,
                "summary_weight": 1, "compute_serialized": True,
                "active_lane": ("summary" if self.turn_owner.summary else "foreground") if self.turn_owner else None,
                "foreground_turns": self.owners[False].turns if self.owners[False] else 0,
                "summary_turns": self.owners[True].turns if self.owners[True] else 0,
                "recovery_required": self.poison_reason is not None,
            }


class RequestValue:
    """A request's progress/statistics never read another request's latest token.

    Outside a scoped observation, monitoring continues to see the latest shared
    value. The descriptors are inert for legacy engines without a scope.
    """
    def __set_name__(self, owner, name):
        self.name = name
        self.shared = "_priority_shared_" + name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        local = getattr(obj, "_priority_local", None)
        values = getattr(local, "values", None)
        if values is not None:
            return values.get(self.name)
        return obj.__dict__.get(self.shared)

    def __set__(self, obj, value):
        local = getattr(obj, "_priority_local", None)
        values = getattr(local, "values", None)
        if values is not None:
            values[self.name] = value
        obj.__dict__[self.shared] = value


@contextlib.contextmanager
def observation(engine):
    # Initialize once before serving; fallback is for CPU fake engines.
    if not hasattr(engine, "_priority_local"):
        engine._priority_local = threading.local()
    local = engine._priority_local
    old = getattr(local, "values", None)
    local.values = {"last": {}, "progress": None, "progress_ms": 0,
                    "reused": 0, "prefill_tok_s_mean": None, "lp_lines": None}
    try:
        yield
    finally:
        local.values = old
