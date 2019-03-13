from dataclasses import dataclass
from urllib.parse import urlparse

from boto3 import client
from botocore.exceptions import ClientError
from typing import Dict


@dataclass
class S3Object:
    bucket: str
    key: str

    @property
    def path(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def exists(self) -> bool:
        try:
            s3 = client("s3")
            s3.head_object(Bucket=self.bucket, Key=self.key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise
        else:
            return True

    @property
    def tags(self) -> Dict[str, str]:
        s3 = client("s3")

        response = s3.get_object_tagging(
            Bucket=self.bucket,
            Key=self.key,
        )

        return {
            tag["Key"]: tag["Value"]
            for tag in response["TagSet"]
        }

    @classmethod
    def from_url(cls, url):
        parsed = urlparse(url)
        return cls(bucket=parsed.netloc, key=parsed.path)
