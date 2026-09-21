"""Regression test for a Connection._listener() infinite-spin bug.

_listener()'s catch-all branch was:

    except Exception as e:
        logger.error(f"background listener error: {e}", exc_info=True)

with no ``break``/``return`` -- so once ``await ws.recv()`` starts raising an
*unrecoverable* generic exception (anything other than
``websockets.exceptions.ConnectionClosed`` or ``asyncio.CancelledError``), the
``while True`` loop immediately retries, gets the same exception again, logs
again, and repeats -- a zero-delay infinite loop that floods the log and
consumes CPU forever instead of ever giving up.

Reproduced in production as: once ``Connection.aclose()`` is scheduled as a
task but the caller closes the event loop before it runs (Browser.stop()
only schedules aclose(); the caller here was googlevoice's
BrowserSender.close()), ``self.get_waiter`` in
websockets' Assembler is left non-None, and every subsequent ``ws.recv()``
call immediately re-raises ``AssertionError: cannot call get() concurrently``
-- observed in production as 384,052 occurrences of that exact assertion
in under 90 seconds before the process was force-killed.

This test reproduces the mechanism directly against the real
``Connection._listener()`` with only the websocket transport stubbed, so it
runs without a browser and is deterministic. It bounds the stub's raise count
so a still-broken implementation fails fast (finite extra iterations observed)
rather than actually hanging the test suite.
"""
import asyncio

import nodriver.core.connection as connection_module


def test_listener_terminates_on_unrecoverable_recv_error(monkeypatch):
    RAISE_BUDGET = 50  # generous ceiling; a fixed listener stops after 1

    class UnrecoverableError(Exception):
        pass

    class FakeSocket:
        close_code = None

        def __init__(self):
            self.recv_calls = 0

        async def recv(self):
            self.recv_calls += 1
            if self.recv_calls > RAISE_BUDGET:
                # Safety valve: if the listener is still spinning after
                # RAISE_BUDGET attempts, stop feeding it and let the test
                # fail on the call-count assertion below instead of hanging.
                raise asyncio.CancelledError()
            raise UnrecoverableError("cannot call get() concurrently")

    sock = FakeSocket()
    conn = connection_module.Connection()
    conn.socket = sock

    failed_with = []
    orig_fail = conn._fail_pending_futures

    def spy_fail(exc):
        failed_with.append(exc)
        orig_fail(exc)

    monkeypatch.setattr(conn, "_fail_pending_futures", spy_fail)

    asyncio.run(conn._listener())

    assert sock.recv_calls == 1, (
        "a single unrecoverable recv() error must stop the listener "
        f"immediately, but ws.recv() was called {sock.recv_calls} times "
        "(the catch-all handler is still looping instead of terminating)"
    )
    assert len(failed_with) == 1 and isinstance(failed_with[0], UnrecoverableError), (
        "the listener must fail pending futures with the triggering "
        f"exception exactly once; got {failed_with}"
    )


def test_listener_survives_a_bad_message_and_still_stops_on_close(monkeypatch):
    """The other half of the contract: a failure while *processing* a message
    that was received fine is not a broken transport, so the listener logs it
    and keeps reading. It then still terminates normally when the socket closes."""
    import websockets

    class FakeSocket:
        close_code = None

        def __init__(self):
            self.recv_calls = 0

        async def recv(self):
            self.recv_calls += 1
            if self.recv_calls == 1:
                return "this is not json"  # received fine, fails to process
            raise websockets.exceptions.ConnectionClosed(None, None)

    sock = FakeSocket()
    conn = connection_module.Connection()
    conn.socket = sock

    asyncio.run(conn._listener())

    assert sock.recv_calls == 2, (
        "a message-processing error must not stop the listener, and a "
        f"subsequent close must; recv() was called {sock.recv_calls} times"
    )
