import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# params: app name, broker url to pull jobs, backend url to store results, and list of modules to import tasks from
celery = Celery(
    "harmonynet",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["api.tasks"],
)

# params to configure how tasks are serialized, how results are stored, and other options
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True, # track when tasks start running rather than just when they are sent to the worker
    result_expires=3600,  # redis auto deletes stored results to prevent unbounded memory growth 
)
