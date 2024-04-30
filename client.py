# from mlserver import MLModel
# import mlserver
# from mlserver.server
import numpy as np
import tritonclient.http as httpclient
# import mlserver

features = {'driver_id': [101], 'customer_id': [2], 'conv_rate': [0.07080516219139099], 'avg_daily_trips': [103], 'acc_rate': [0.32292142510414124], 'lifetime_trip_count': [862], 'avg_passenger_count': [0.833490788936615]}

triton_client = httpclient.InferenceServerClient(url='host.docker.internal:8000')
print(triton_client.is_server_live())
metadata = triton_client.get_model_metadata('driver-success-model-onnx')
print(metadata)

# inputs = []

# type_map = {
#     'FP32': np.float32,
#     'INT64': np.int64
# }

# for model_input in metadata['inputs']:
#     ii = httpclient.InferInput(model_input['name'], shape=[1, 1], datatype=model_input['datatype'])
#     ar = np.array(features[model_input['name']], type_map[model_input['datatype']])
#     ar = ar.reshape((1, 1))
#     ii.set_data_from_numpy(ar)
#     inputs.append(ii)
#     # inputs.append(httpclient.InferInput(model_input['name'], shape=[-1, 1], datatype=model_input['datatype']))

# res = triton_client.infer('driver-success-model-onnx', inputs=inputs)
# print(res.as_numpy('probabilities'))


# inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
# inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))

# # Initialize the data
# inputs[0].set_data_from_numpy(input0_data, binary_data=False)
# inputs[1].set_data_from_numpy(input1_data, binary_data=True)

triton_client = httpclient.InferenceServerClient(url='localhost:8080')
print(triton_client.is_server_live())
metadata = triton_client.get_model_metadata('driver-success-model-onnx')

print(triton_client.infer(model_name='driver-success-model-onnx', inputs=[]).as_numpy('probabilities'))



# # inputs = []
# #     outputs = []
# #     inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))
# #     inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))

# #     # Create the data for the two input tensors. Initialize the first
# #     # to unique integers and the second to all ones.
# #     input0_data = np.arange(start=0, stop=16, dtype=np.int32)
# #     input0_data = np.expand_dims(input0_data, axis=0)
# #     input1_data = np.ones(shape=(1, 16), dtype=np.int32)

# #     # Initialize the data
# #     inputs[0].set_data_from_numpy(input0_data)
# #     inputs[1].set_data_from_numpy(input1_data)

# #     outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
# #     outputs.append(grpcclient.InferRequestedOutput("OUTPUT1"))

# #     # Test with outputs
# #     results = triton_client.infer(
# #         model_name=model_name,
# #         inputs=inputs,
# #         outputs=outputs,
# #         client_timeout=FLAGS.client_timeout,
#         headers={"test": "1"},
#         compression_algorithm=FLAGS.grpc_compression_algorithm,
#     )