from feast import FeatureStore, FeatureService
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from onnx_helper import to_oonx_auto
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

store = FeatureStore(repo_path='feature_repo')


entity_df = pd.read_parquet('data/orders.parquet')

feature_service = store.get_feature_service('driver_success')

features = [f.name for fvp in feature_service.feature_view_projections for f in fvp.features]

training_set = store.get_historical_features(
    entity_df=entity_df,
    features=feature_service
).to_df()

training_set['avg_daily_trips'] = pd.to_numeric(training_set['avg_daily_trips'], errors="coerce").astype('int64')
training_set['lifetime_trip_count'] = pd.to_numeric(training_set['lifetime_trip_count'], errors="coerce").astype('int64')

X = training_set[features]
y = training_set["order_is_success"]

remote_server_uri = "http://mlflow:8080"
mlflow.set_tracking_uri(remote_server_uri)
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

    print(pipe.decision_function(x_test))

    signature = infer_signature(x_test, y_pred)

    onnx_model = to_oonx_auto(pipe, x_test, check_conversion=False, check_sample_size=1000)

    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="model-onnx",
        signature=signature,
        registered_model_name="driver-success-model-onnx",
    )

    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model-sklearn",
        signature=signature,
        registered_model_name="driver-success-model-sklearn",
    )    
