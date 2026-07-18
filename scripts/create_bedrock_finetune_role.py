"""
Create IAM role for Bedrock Model Customization (fine-tuning).

Run ONCE before submitting the fine-tuning job:
    python scripts/create_bedrock_finetune_role.py

Required permissions for the IAM user running this:
  iam:CreateRole, iam:AttachRolePolicy, iam:PutRolePolicy

The created role allows:
  - bedrock.amazonaws.com to assume it (trust policy)
  - Read from v1-gordonai/agentcommercebench/finetune/ (training data)
  - Write to v1-gordonai/agentcommercebench/finetune/output/ (model artifacts)
"""
import boto3, json, os
from dotenv import load_dotenv
load_dotenv()

ROLE_NAME   = "BedrockFineTuneRole"
BUCKET      = "v1-gordonai"
S3_PREFIX   = "agentcommercebench/finetune"
ACCOUNT_ID  = "170554564926"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {
            "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
            "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:us-east-1:{ACCOUNT_ID}:model-customization-job/*"},
        },
    }],
}

INLINE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3TrainingDataAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{BUCKET}",
                f"arn:aws:s3:::{BUCKET}/{S3_PREFIX}/*",
            ],
        },
        {
            "Sid": "S3OutputAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{BUCKET}",
                f"arn:aws:s3:::{BUCKET}/{S3_PREFIX}/output/*",
            ],
        },
    ],
}


def create_role():
    iam = boto3.client(
        "iam",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="Allows Bedrock to access S3 for AgentCommerceBench fine-tuning",
        )
        role_arn = resp["Role"]["Arn"]
        print(f"Created role: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"
        print(f"Role already exists: {role_arn}")

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="BedrockFineTuneS3Access",
        PolicyDocument=json.dumps(INLINE_POLICY),
    )
    print(f"Attached inline policy: BedrockFineTuneS3Access")
    print(f"\nRole ARN: {role_arn}")
    print(f"Now run: python -m benchmark.models.bedrock_finetune")
    return role_arn


if __name__ == "__main__":
    create_role()
