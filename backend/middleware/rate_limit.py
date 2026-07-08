import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory token bucket rate limiter per IP.

    100 requests/minute per client by default.
    Configure via RAG_RATE_LIMIT env or the limit/interval params.
    """

    def __init__(self, app, limit: int = 100, interval_seconds: int = 60):
        super().__init__(app)
        self.limit = limit
        self.interval = interval_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for static files and health checks
        path = request.url.path
        if path.startswith(("/static/", "/assets/", "/api/health", "/api/admin/health")):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.interval
        bucket = self._buckets[client_ip]

        # Prune old entries
        self._buckets[client_ip] = [t for t in bucket if t > cutoff]

        if len(self._buckets[client_ip]) >= self.limit:
            raise HTTPException(429, detail="Too many requests. Try again shortly.")

        self._buckets[client_ip].append(now)
        return await call_next(request)
