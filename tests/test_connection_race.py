"""Regression test for a Connection.aopen() concurrency bug.

Before this fix, aopen()'s socket-state check, websockets.connect(), and
listener-task creation were not one atomic critical section:

    if not self.socket or bool(self.socket.close_code):
        self.socket = await websockets.connect(...)
        self._listener_task = asyncio.create_task(self._listener())

Two coroutines racing to attach() the same Connection (this happens in
practice via Browser.get()'s implicit attach() through .send() racing its
own explicit follow-up attach(), or via a concurrent auto-attach event
handler reacting to target/session churn during navigation) could both
observe "no socket yet" before either had assigned self.socket, each open
their own websocket, and each spawn a listener task. Because _listener()
reads ``self.socket`` only once it actually starts running (not when the
task is created), both listener tasks could end up bound to whichever
socket was assigned last -- two concurrent ``await ws.recv()`` calls on one
connection, surfacing to users as:

    AssertionError: cannot call get() concurrently

(raised from websockets' Assembler.get(), logged by _listener()'s own
``except Exception`` handler as "background listener error: ...").

Reproduced against a real, previously-used Chrome profile navigating
cross-origin (no special setup needed to trigger it -- just repeated
launches against the same profile_dir navigating to a site with any
redirect/target churn). This test reproduces the mechanism directly against
the real Connection class with only the websocket transport stubbed, so it
runs without a browser and is deterministic.
"""
import asyncio

import nodriver.core.connection as connection_module


def test_concurrent_attach_opens_exactly_one_socket_and_listener(monkeypatch):
    created_sockets = []

    class FakeSocket:
        def __init__(self):
            self.close_code = None
            self.id = len(created_sockets)

    async def fake_connect(url, **kwargs):
        # Yield control, like a real handshake does, so a second concurrent
        # aopen() call would (without the fix) also pass the
        # "not self.socket" guard before the first has assigned it.
        await asyncio.sleep(0)
        sock = FakeSocket()
        created_sockets.append(sock)
        return sock

    listener_bound_to = []

    async def fake_listener(self):
        # Mirrors the real _listener()'s first line, `ws = self.socket`,
        # read only once the task actually gets scheduled to run.
        await asyncio.sleep(0)
        listener_bound_to.append(self.socket.id)

    monkeypatch.setattr(connection_module.websockets, "connect", fake_connect)
    monkeypatch.setattr(connection_module.Connection, "_listener", fake_listener)

    conn = connection_module.Connection()
    conn.websocket_url = "ws://fake/devtools/browser/x"

    async def drive():
        await asyncio.gather(conn.aopen(), conn.aopen())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(drive())

    assert len(created_sockets) == 1, (
        "two simultaneous aopen() calls must open exactly one socket, got "
        f"{len(created_sockets)}"
    )
    assert listener_bound_to == [0], (
        "exactly one listener task must run, bound to the one socket that "
        f"was opened; got {listener_bound_to}"
    )
