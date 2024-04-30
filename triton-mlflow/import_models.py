import mlflow
import numpy as np
from mlflow import MlflowClient
import shutil
import os
import yaml

with open('triton-mlflow.yaml', "r") as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

remote_server_uri = os.environ.get('MLFLOW_URL', "http://mlflow:8080")  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)

import onnx

base_dir = 'models'

os.mkdir(base_dir)

for model in conf['models']:
    model_name = model['name']
    model_path = os.path.join(base_dir, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    for version in model['versions']:
        model_version = version['version']
        registered_model_name = version['registered_model_name']
        registered_model_version = version['registered_model_version']
        version_path = os.path.join(base_dir, model_name, model_version)
        if os.path.exists(version_path):
            shutil.rmtree(version_path)
        os.mkdir(version_path)
        onnx_model = mlflow.onnx.load_model(model_uri=f"models:/{registered_model_name}/{registered_model_version}")
        onnx.save(onnx_model, os.path.join(version_path, 'model.onnx'))
