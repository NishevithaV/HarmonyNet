import os
from celery import Celery

REDIS_URL = (
    os.environ.get("UPSTASH_REDIS_URL") or
    os.environ.get("REDIS_URL") or
    "redis://localhost:6379/0"
)

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
