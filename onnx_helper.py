from skl2onnx import to_onnx, update_registered_converter
import numpy
from skl2onnx.common.shape_calculator import (calculate_linear_classifier_output_shapes, calculate_linear_regressor_output_shapes)
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from typing import List
from feast.types import Invalid, Bytes, String, Bool, Int32, Int64, Float32, Float64, UnixTimestamp
from onnxconverter_common.data_types import FloatTensorType, Int64TensorType

def to_onnx_schema(features: List):
    type_map = {
        Float32: FloatTensorType,
        Int64: Int64TensorType
    }

    return [
        (f.name, type_map[f.dtype](shape=[None, 1]))
        for f in features
    ]


def to_onnx_auto(sklearn_pipeline, schema):
    steps = [step.__class__ for step in sklearn_pipeline]

    # import xgboost as xgb
    # if xgb.sklearn.XGBClassifier in steps:
    #     update_registered_converter(
    #         xgb.XGBClassifier, 'XGBoostXGBClassifier',
    #         calculate_linear_classifier_output_shapes, convert_xgboost,
    #         options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

    from sklearn.linear_model import LogisticRegression

    onnx_model = to_onnx(sklearn_pipeline, initial_types=schema, options = {LogisticRegression: {'zipmap': False}})
    return onnx_model

def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    # print(numpy.sort(d))
    return d.max(), (d / numpy.abs(p1)).max()

def compare_models(sklearn_pipeline, onnx_model, X, predict_method="predict_proba"):
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    inputs = {c: X[c].values.reshape((-1, 1)) for c in X.columns}

    onnx_out = sess.run(None, inputs)[1]

    fn_predict = getattr(sklearn_pipeline, predict_method)
    sklearn_out = fn_predict(X)
    max_diff, _ = diff(onnx_out, sklearn_out)

    if max_diff > 0.0000001:
        print(onnx_out)
        print(sklearn_out)

        raise Exception(f'ONNX and Sklearn models are not identical, max diff value -> {max_diff}')
