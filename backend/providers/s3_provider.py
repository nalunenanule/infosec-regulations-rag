import boto3

class S3Provider:
    """
    Класс-провайдер для работы с S3-хранилищем.
    """
    
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key

    def get_s3_client(self):
        return boto3.client(
            "s3",
            region_name="us-east-1",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
