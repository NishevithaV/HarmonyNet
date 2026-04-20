import os
import boto3
from botocore.config import Config

_BUCKET = os.environ.get("R2_BUCKET_NAME", "harmonynet")


def _client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload(local_path: str, key: str) -> None:
    _client().upload_file(local_path, _BUCKET, key)


def presigned_url(key: str, filename: str, content_type: str, expires: int = 3600) -> str:
    return _client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": _BUCKET,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{filename}"',
            "ResponseContentType": content_type,
        },
        ExpiresIn=expires,
    )
