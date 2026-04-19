import os
import ssl
from celery import Celery

_upstash = os.environ.get("UPSTASH_REDIS_URL", "")
_redis = os.environ.get("REDIS_URL", "")

REDIS_URL = _upstash or _redis or "redis://localhost:6379/0"

print(f"[celery_app] UPSTASH_REDIS_URL={'set' if _upstash else 'MISSING'}", flush=True)
print(f"[celery_app] REDIS_URL={'set len={}'.format(len(_redis)) if _redis else 'empty/missing'}", flush=True)
print(f"[celery_app] Using broker: {REDIS_URL[:40]}...", flush=True)

# Create app without broker/backend so CLI auto_envvar_prefix can't override them
celery = Celery("harmonynet", include=["api.tasks"])

# Set broker and backend directly on conf — this takes priority over CLI env vars
_ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE} if REDIS_URL.startswith("rediss://") else {}

celery.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    broker_use_ssl=_ssl_opts or None,
    redis_backend_use_ssl=_ssl_opts or None,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)
