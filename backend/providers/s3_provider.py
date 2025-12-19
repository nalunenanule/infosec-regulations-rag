import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import List
import io

class S3Provider:
    """
    Класс-провайдер для работы с S3-хранилищем.
    Позволяет получать список файлов и загружать их в память.
    """
    
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            file_list = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    file_list.append(obj["Key"])
            return file_list
        except (BotoCoreError, ClientError) as e:
            print(f"Ошибка при получении списка файлов: {e}")
            return []

    def get_file_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Генерирует временную ссылку на объект S3/MinIO
        """
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expires_in
        )

    def get_files_urls(self, prefix: str = "", expires_in: int = 3600) -> List[str]:
        urls = []
        for key in self.list_files(prefix):
            urls.append(self.get_file_url(key, expires_in=expires_in))
        return urls
