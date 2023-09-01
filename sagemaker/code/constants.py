
import boto3
import sagemaker
from pathlib import Path

BUCKET = "default-soil-predictions"
S3_LOCATION = f"s3://{BUCKET}/soilpreds"

DATA_ACCUWEATHER_FILEPATH = Path().resolve() / "data" / "accuweather_hourly_1.29_to_6.15.csv"
DATA_WEATHERLINK_COMPARE_FILEPATH = Path().resolve() / "data" / "meteo_data for model 30.1.2023. - 31.7.2023..csv"
DATA_WEATHERLINK_MODEL_FILEPATH = Path().resolve() / "data" / "meteo_data for model 30.1.2023. - 31.7.2023..csv"
DATA_SENSOR1_FILEPATH = Path().resolve() / "data" / "meteo_data for model 30.1.2023. - 31.7.2023..csv"
DATA_SENSOR2_FILEPATH = Path().resolve() / "data" / "meteo_data for model 30.1.2023. - 31.7.2023..csv"


sagemaker_client = boto3.client("sagemaker")
iam_client = boto3.client("iam")
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
