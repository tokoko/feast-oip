import pandas as pd
import mlflow
from feast import FeatureStore, FeatureService
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature
from onnx_helper import to_onnx_auto, to_onnx_schema, compare_models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

store = FeatureStore(repo_path='feature_repo')

entity_df = pd.read_parquet('data/orders.parquet')

feature_service = store.get_feature_service('driver_success')

training_set = store.get_historical_features(
    entity_df=entity_df,
    features=feature_service
).to_df()

training_set['avg_daily_trips'] = pd.to_numeric(training_set['avg_daily_trips'], errors="coerce").astype('int64')
training_set['lifetime_trip_count'] = pd.to_numeric(training_set['lifetime_trip_count'], errors="coerce").astype('int64')

features = [f for fvp in feature_service.feature_view_projections for f in fvp.features]

X = training_set[[f.name for f in features]]
y = training_set["order_is_success"]

mlflow.set_tracking_uri("http://mlflow:8080")
mlflow.set_experiment("/my-experiment")

with mlflow.start_run():
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    penalty="l2"
    logisticRegr = LogisticRegression(penalty=penalty)

    pipe = Pipeline(
        [
            ('preprocess', StandardScaler()),
            ('lr', logisticRegr)
        ]
    )

    mlflow.log_param("penalty", penalty)
    pipe.fit(x_train, y_train)

    score = pipe.score(x_test, y_test)
    mlflow.log_metric("score", score)

    y_pred = pipe.predict_proba(x_test)
    signature = infer_signature(x_test, y_pred)

    onnx_input_schema = to_onnx_schema(features=features)
    onnx_model = to_onnx_auto(pipe, onnx_input_schema)

    compare_models(pipe, onnx_model, x_test)

    # mlflow.onnx.log_model(
    #     onnx_model=onnx_model,
    #     artifact_path="model-onnx",
    #     signature=signature,
    #     registered_model_name="driver-success-model-onnx",
    # )

    # mlflow.sklearn.log_model(
    #     sk_model=pipe,
    #     artifact_path="model-sklearn",
    #     signature=signature,
    #     registered_model_name="driver-success-model-sklearn",
    # )
