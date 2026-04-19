import os
import ssl
from celery import Celery

_upstash = os.environ.get("UPSTASH_REDIS_URL", "")
_redis = os.environ.get("REDIS_URL", "")

REDIS_URL = _upstash or _redis or "redis://localhost:6379/0"

# rediss:// (Upstash SSL) requires ssl_cert_reqs both in the URL params
# and in the broker/backend SSL conf — set both so nothing falls through.
_is_ssl = REDIS_URL.startswith("rediss://")
if _is_ssl and "ssl_cert_reqs" not in REDIS_URL:
    REDIS_URL += "?ssl_cert_reqs=CERT_NONE"

_ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE} if _is_ssl else None

print(f"[celery_app] UPSTASH_REDIS_URL={'set' if _upstash else 'MISSING'}", flush=True)
print(f"[celery_app] REDIS_URL={'set' if _redis else 'empty/missing'}", flush=True)
print(f"[celery_app] Using broker: {REDIS_URL[:60]}...", flush=True)

celery = Celery("harmonynet", include=["api.tasks"])

celery.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    broker_use_ssl=_ssl_opts,
    redis_backend_use_ssl=_ssl_opts,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)
