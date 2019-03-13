"""
Commands for interacting with AWS
"""

from json import dumps, load
from hashlib import sha1
from io import BytesIO
from copy import copy
from os.path import join

from click import command, option, File, echo, style, Path
from boto3 import client

from microcosm_sagemaker.s3 import S3Object
from microcosm_sagemaker.constants import CONFIGURATION_TAG_KEY


@command()
@option(
    "--configuration",
    type=File("rb"),
    required=True
)
@option(
    "--bucket",
    type=str,
    required=True
)
@option(
    "--model_name",
    type=str,
    required=True
)
@option(
    "--tag",
    type=str,
    required=False,
)
def put_sagemaker_config(configuration, bucket, model_name, tag):
    configuration_flat = dumps(load(configuration), separators=(",", ":"))
    configuration_hash = sha1(configuration_flat.encode()).hexdigest()

    remote_path = S3Object(bucket, f"{model_name}/{configuration_hash}.json")

    if remote_path.exists:
        echo(style(f"Configuration already uploaded.", fg="red"))
        echo(f"Existing URL: {remote_path.path}")
        return

    configuration.seek(0)

    s3 = client("s3")
    s3.upload_fileobj(configuration, remote_path.bucket, remote_path.key)

    if tag:
         s3.put_object_tagging(
            Bucket=remote_path.bucket,
            Key=remote_path.key,
            Tagging={
                "TagSet": [
                    {
                        "Key": CONFIGURATION_TAG_KEY,
                        "Value": tag
                    },
                ],
            },
        )

    echo(style(f"Configuration uploaded successfully.", fg="green"))
    echo(f"Configuration URL: {remote_path.path}")


@command()
@option(
    "--bucket",
    type=str,
    required=True
)
@option(
    "--model_name",
    type=str,
    required=True
)
def list_sagemaker_configs(bucket, model_name):
    s3 = client("s3")

    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=model_name,
    )

    configuration_files = filter(
        lambda entity: entity["Size"] > 0 and entity["Key"].endswith(".json"),
        response["Contents"],
    )

    configuration_files = sorted(
        configuration_files,
        key=lambda entity: entity["LastModified"],
        reverse=True,
    )

    configuration_objects = map(
        lambda entity: S3Object(bucket, entity["Key"]),
        configuration_files,
    )

    for object in configuration_objects:
        config_tag = object.tags[CONFIGURATION_TAG_KEY]
        echo(f"{config_tag} : {object.path}")


@command()
@option(
    "--bucket",
    type=str,
    required=True
)
@option(
    "--model_name",
    type=str,
    required=True
)
@option(
    "--sha",
    type=str,
    required=True,
)
@option(
    "--local",
    type=Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
def get_sagemaker_config(bucket, model_name, sha, local):
    object = S3Object(bucket, f"{model_name}/{sha}.json")
    local_path = join(local, f"{sha}.json")

    if not object.exists:
        echo(style("Configuration not found on S3.", fg="red"))
        return

    s3 = client("s3")
    s3.download_file(object.bucket, object.key, local_path)

    echo(style(f"Configuration downloaded successfully.", fg="green"))
    echo(f"Local configuration: {local_path}")
