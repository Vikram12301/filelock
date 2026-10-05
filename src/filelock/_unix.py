from __future__ import annotations

import os
import sys
import threading
import warnings
from contextlib import suppress
from errno import EACCES, EAGAIN, ENOSYS, EWOULDBLOCK
from pathlib import Path
from typing import Final, cast

from ._api import BaseFileLock
from ._util import ensure_directory_exists

has_fcntl = False
if sys.platform == "win32":  # pragma: win32 cover

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError

else:  # pragma: win32 no cover
    try:
        import fcntl

        _ = (fcntl.flock, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN)
    except (ImportError, AttributeError):
        _FCNTL_UNAVAILABLE: Final[str] = "fcntl is unavailable"

        def _lock_fd_nonblocking(_fd: int) -> bool:
            raise OSError(ENOSYS, _FCNTL_UNAVAILABLE)

        def _unlock_fd(_fd: int) -> None:
            raise OSError(ENOSYS, _FCNTL_UNAVAILABLE)

    else:
        has_fcntl = True
        # Contention errnos for a nonblocking flock. EAGAIN/EWOULDBLOCK are the usual "held elsewhere" codes; some
        # filesystems report EACCES instead, so treat it as contention too rather than a permanent error.
        _CONTENTION_ERRNOS: Final[frozenset[int]] = frozenset({EACCES, EAGAIN, EWOULDBLOCK})

        def _lock_fd_nonblocking(fd: int) -> bool:
            # One nonblocking exclusive flock attempt shared by UnixFileLock and lock_descriptor, so both contend on
            # the same lock and classify errors identically. The caller owns fd; this never closes it.
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exception:
                if exception.errno in _CONTENTION_ERRNOS:
                    return False
                raise
            return True

        def _unlock_fd(fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_UN)

        # One gate per lock file path serializes this process's threads in front of flock(), so two threads of one
        # process never hold the same path at once, whichever way the platform's flock() treats descriptors. Keyed
        # by path so a lock file that does not exist yet is still gated.
        class _IntraProcessGates:
            def __init__(self) -> None:
                self.reset()

            def reset(self) -> None:
                self.guard = threading.Lock()
                self.by_path: dict[str, threading.Lock] = {}

            def gate(self, path: str) -> threading.Lock:
                with self.guard:
                    gate = self.by_path.get(path)
                    if gate is None:
                        gate = self.by_path[path] = threading.Lock()
                    return gate

        _intra_process_gates = _IntraProcessGates()

        def _intra_process_gate(path: str) -> threading.Lock:
            return _intra_process_gates.gate(path)

        def _reset_intra_process_gates_in_child() -> None:  # pragma: forked child
            # A forked child inherits each gate in whatever state the parent's threads left it. A gate held by a thread
            # that exists only in the parent would never open, so the child starts with fresh, unheld gates.
            _intra_process_gates.reset()

        if hasattr(os, "register_at_fork"):  # pragma: needs fork
            os.register_at_fork(after_in_child=_reset_intra_process_gates_in_child)

    class UnixFileLock(BaseFileLock):
        """
        Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.

        We leave the lock file in place after release. Unlinking a locked file on Unix splits
        waiters across inodes and breaks mutual exclusion for processes that coordinate via the
        same path.
        """

        def _acquire(self) -> None:
            gate = _intra_process_gate(self.lock_file)
            if not gate.acquire(blocking=False):
                return  # a sibling thread in this process is already attempting or holding this lock file
            keep_gate = False
            try:
                missing_flock = self._acquire_native()
                if missing_flock is not None:
                    self._switch_to_soft_lock(*missing_flock)
                else:
                    keep_gate = self._context.lock_file_fd is not None
            finally:
                if not keep_gate:
                    gate.release()

        def _acquire_native(self) -> tuple[int, OSError] | None:
            ensure_directory_exists(self.lock_file)
            # Open without O_TRUNC and defer truncation and fchmod until after flock succeeds: a contender that loses
            # the lock must not truncate the holder's file (erasing caller diagnostics) or change its mode. The winner
            # truncates and normalizes mode once it owns the lock (#591).
            open_flags = os.O_RDWR
            if (o_nofollow := getattr(os, "O_NOFOLLOW", None)) is not None:
                open_flags |= o_nofollow
            open_flags |= os.O_CREAT
            open_mode = self._open_mode()
            try:
                fd = os.open(self.lock_file, open_flags, open_mode)
            except FileNotFoundError:
                # On FUSE/NFS, os.open(O_CREAT) is not atomic; a split LOOKUP + CREATE lets a concurrent unlink()
                # delete the file between them. For a valid path, treat ENOENT as transient contention. For an
                # invalid path (e.g. empty string), re-raise to avoid an infinite retry loop.
                if self.lock_file and Path(self.lock_file).parent.exists():
                    return None
                raise
            except PermissionError:
                # Sticky-bit dirs (e.g. /tmp): O_CREAT fails if the file is owned by another user (#317).
                # Fall back to opening the existing file without O_CREAT.
                if not Path(self.lock_file).exists():
                    raise
                try:
                    fd = os.open(self.lock_file, open_flags & ~os.O_CREAT, open_mode)
                except FileNotFoundError:
                    return None
            self._mark_descriptor_pending(fd)
            try:
                locked = _lock_fd_nonblocking(fd)
            except OSError as exception:
                if exception.errno != ENOSYS:
                    self._mark_descriptor_released()
                    os.close(fd)
                    raise  # contention returns False from _lock_fd_nonblocking, so any raise here is a real failure
                return fd, exception
            if locked:
                self._finalize_locked_fd(fd)
            else:
                self._mark_descriptor_released()
                os.close(fd)  # contention; let the retry loop try again
            return None

        def _switch_to_soft_lock(self, fd: int, missing_flock: OSError) -> None:
            # The filesystem does not implement flock. Capture the opened file's identity before closing so the cleanup
            # below removes only this attempt's placeholder, not a peer's replacement.
            identity: tuple[int, int] | None = None
            with suppress(OSError):
                identity = (fstat := os.fstat(fd)).st_dev, fstat.st_ino
            self._mark_descriptor_released()
            os.close(fd)
            if not self._fallback_to_soft or self._preserve_lock_file or self._on_acquired is not None:
                # Fail closed: the caller opted out of existence-lock semantics (#603), asked to preserve the pathname
                # (#605), or set an on_acquired hook (#607), none of which a soft lock can honor.
                raise missing_flock
            with suppress(OSError):
                current = os.lstat(self.lock_file)
                if identity == (current.st_dev, current.st_ino):
                    Path(self.lock_file).unlink()
            self._fallback_to_soft_lock()
            self._acquire()

        def _finalize_locked_fd(self, fd: int) -> None:
            # Runs with the flock held. Truncate and normalize mode under a guard so any failure closes fd rather than
            # leaking it and its lock. A concurrent _release() may have unlinked the inode between our open() and
            # flock() (st_nlink 0), leaving a useless dead-inode lock; drop it and let the retry loop start fresh.
            keep = False
            try:
                stat_result = os.fstat(fd)
                if stat_result.st_nlink != 0:
                    os.ftruncate(fd, 0)
                    self._apply_explicit_mode(fd)
                    keep = True
            except OSError:
                self._mark_descriptor_released()
                os.close(fd)
                raise
            if keep:
                self._mark_descriptor_owned(fd, (stat_result.st_dev, stat_result.st_ino))
            else:
                self._mark_descriptor_released()
                os.close(fd)

        def _apply_explicit_mode(self, fd: int) -> None:
            if self.has_explicit_mode:
                with suppress(PermissionError):
                    os.fchmod(fd, self._context.mode)

        def _fallback_to_soft_lock(self) -> None:
            # Import lazily: this runs only on the rare flock fallback, and asyncio imports _unix, so a
            # module-level import of it here would cycle.
            from ._soft import SoftFileLock  # ruff:ignore[import-outside-top-level]

            warnings.warn("flock not supported on this filesystem, falling back to SoftFileLock", stacklevel=2)
            from .asyncio import AsyncSoftFileLock, BaseAsyncFileLock  # ruff:ignore[import-outside-top-level]

            self.__class__ = AsyncSoftFileLock if isinstance(self, BaseAsyncFileLock) else SoftFileLock

        def _release(self) -> None:
            fd = cast("int", self._context.lock_file_fd)
            # Retain the descriptor until flock succeeds: a failed unlock leaves the kernel lock held, so is_locked
            # must keep reporting held for a retry. Once flock commits, clear held state and close as post-unlock
            # cleanup; a close failure (EIO on FUSE/Docker bind mounts) does not make the kernel lock held again.
            _unlock_fd(fd)
            self._mark_descriptor_released()
            self._close_released_fd(fd, default_suppresses=True)
            # Release the gate last, so a sibling thread never sees it open while the kernel lock is still held.
            _intra_process_gate(self.lock_file).release()


if sys.platform == "win32":  # pragma: win32 cover
    __all__ = ["UnixFileLock", "has_fcntl"]
else:  # pragma: win32 no cover
    __all__ = ["UnixFileLock", "_lock_fd_nonblocking", "_unlock_fd", "has_fcntl"]
