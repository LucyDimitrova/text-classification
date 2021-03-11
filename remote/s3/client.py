import boto3
import pandas as pd
import s3fs
from datetime import *
from botocore.exceptions import ClientError
from memory_profiler import profile

from algorithms.abstract import Model


class Client:
    def __init__(self):
        self.client = boto3.client('s3')

    # @profile
    def load_dataframe(self, path, **kwargs):
        """Load file into a dataframe

        :param path: Path to file in S3
        :return: dataframe if file was fetched, else None
        """
        try:
            print(f'Started reading file {path} at {datetime.now()}')
            df = pd.read_csv(path, **kwargs)
            if 'chunksize' in kwargs.keys():
                df = pd.concat(df)
            print(f'Reading file {path} finished at {datetime.now()}')
            return df
        except ClientError as err:
            print(f'There was an error loading file at {path}: {err}')

    # @profile
    def upload(self, local_path, bucket, remote_path):
        """Upload generic file to S3

        :param local_path: path to file in local system
        :param bucket: S3 bucket
        :param remote_path: path to file in S3 bucket
        :return: Boolean, True if successful, False if error occurred
        """
        try:
            response = self.client.upload_file(local_path, bucket, remote_path)
        except ClientError as e:
            print(f'Error uploading file {local_path} to S3: ', e)
            return False
        return True

    def upload_dataframe(self, df, path):
        """Upload dataframe to S3 bucket

        :param df: Dataframe to upload
        :param path: Path to store dataframe to in S3
        """
        try:
            df.to_csv(path, index=False)
        except Exception as err:
            print(f'There was an error uploading dataframe at {path}: {err}')

    def load_model(self, bucket, path, local_path):
        """Download model from S3 bucket and load it into a model instance

        :param bucket: S3 bucket
        :param path: path to file in S3 bucket
        :param local_path: path to save file to local system
        :return: Model classifier
        """
        try:
            model = Model()
            print(f'Started reading model file {bucket + path} at {datetime.now()}')
            self.client.download_file(bucket, path, local_path)
            print(f'Finished reading model file {bucket + path} at {datetime.now()}')
            return model.load(local_path)
        except Exception as err:
            print(f'There was an error reading model file at {bucket + path}: {err}')


