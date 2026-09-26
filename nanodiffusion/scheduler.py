from collections import deque

from nanodiffusion.request import Request


class RequestScheduler:
    """FIFO admission: one active image request keeps the first implementation small."""

    def __init__(self):
        self.waiting: deque[Request] = deque()
        self.active: Request | None = None

    @property
    def idle(self) -> bool:
        return self.active is None and not self.waiting

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> Request:
        if self.active is None:
            if not self.waiting:
                raise RuntimeError("no queued requests")
            self.active = self.waiting.popleft()
        return self.active

    def finish(self) -> None:
        if self.active is None:
            raise RuntimeError("no active request")
        self.active = None
