import os
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
celery.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)
