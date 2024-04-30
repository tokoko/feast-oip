from skl2onnx import to_onnx, update_registered_converter
import numpy
import xgboost as xgb
from skl2onnx.common.shape_calculator import (calculate_linear_classifier_output_shapes, calculate_linear_regressor_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
# from mlprodict.onnx_conv import guess_schema_from_data
import onnxruntime as rt

def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    # print(numpy.sort(d))
    return d.max(), (d / numpy.abs(p1)).max()

def guess_schema_from_data(X):
    from skl2onnx.algebra.type_helper import guess_initial_types
    init = guess_initial_types(X=X, initial_types=None)
    unique = set()
    for _, col in init:
        if len(col.shape) != 2:
            return init
        if col.shape[0] is not None:
            return init
        if len(unique) > 0 and col.__class__ not in unique:
            return init
        unique.add(col.__class__)
    unique = list(unique)
    return [("X", unique[0]([None, sum(_[1].shape[1] for _ in init)]))]

def to_oonx_auto(sklearn_pipeline, 
                 X, 
                 check_conversion=False,
                 check_sample_size=None,
                 predict_method="predict_proba"
                 ):
    steps = [step.__class__ for step in sklearn_pipeline]

    # if xgb.sklearn.XGBClassifier in steps:
    #     update_registered_converter(
    #         xgb.XGBClassifier, 'XGBoostXGBClassifier',
    #         calculate_linear_classifier_output_shapes, convert_xgboost,
    #         options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

    from onnxconverter_common.data_types import FloatTensorType, Int64TensorType

    schema = [
        ('conv_rate', FloatTensorType(shape=[None, 1])), 
        ('acc_rate', FloatTensorType(shape=[None, 1])), 
        ('avg_daily_trips', Int64TensorType(shape=[None, 1])), 
        ('avg_passenger_count', FloatTensorType(shape=[None, 1])), 
        ('lifetime_trip_count', Int64TensorType(shape=[None, 1]))
    ]

    from sklearn.linear_model import LogisticRegression

    onnx_model = to_onnx(sklearn_pipeline, initial_types=schema, options = {LogisticRegression: {"raw_scores": True, 'zipmap': False}, xgb.XGBClassifier: {'zipmap': False}})

    if check_conversion:
        sess = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
        inputs = {c: X[c].values.reshape((-1, 1)) for c in X.columns}
        onnx_out = sess.run(None, inputs)[1]

        if check_sample_size:
            onnx_out = onnx_out[0:check_sample_size]

        fn_predict = getattr(sklearn_pipeline, predict_method)
        sklearn_out = fn_predict(X)

        if check_sample_size:
            sklearn_out = sklearn_out[0:check_sample_size]

        print(onnx_out)
        print(sklearn_out)

        max_diff, _ = diff(onnx_out, sklearn_out)
        print(max_diff)

        if max_diff > 0.00001:
            print(max_diff)
            # print(onnx_out)
            # print(sklearn_out)
        # print(type(onnx_out))
        # print(type(sklearn_out))

    return onnx_model
