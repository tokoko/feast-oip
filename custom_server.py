from mlserver.codecs import decode_args
from mlserver import MLModel
from typing import List
import numpy as np
from feast import FeatureStore, feature_view_projection
from mlserver.codecs.numpy import NumpyCodec

from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    RequestOutput,
    ResponseOutput,
    MetadataModelResponse,
    MetadataTensor,
    TensorData
)

from tritonclient.http._infer_input import InferInput
import tritonclient.http as httpclient

class MyKulModel(MLModel):

    async def load(self):
        print(self._settings)
        self.fs = FeatureStore('feature_repo')
        self.feature_service = self.fs.get_feature_service(self._settings.parameters.feature_service)

        feature_view_names = [fvp.name for fvp in self.feature_service.feature_view_projections]
        for fv in feature_view_names:
            entity_columns = self.fs.get_feature_view(fv).entity_columns
            self._settings.inputs.extend(
                [MetadataTensor(name=f.name, datatype="INT64", shape=[-1, 1]) for f in entity_columns]
            )


        self.triton_client = httpclient.InferenceServerClient(url=self._settings.parameters.oip_url)
        self._metadata = self.triton_client.get_model_metadata(self._settings.parameters.oip_model)
        
        for out in self._metadata['outputs']:
            self._settings.outputs.append(
                MetadataTensor(name=out['name'], datatype=out['datatype'], shape=out['shape'])
            )

    
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        self.fs = FeatureStore('feature_repo')
        self.feature_service = self.fs.get_feature_service(self._settings.parameters.feature_service)
        features = self.fs.get_online_features(
            self.feature_service,
            [
                {"driver_id": 101, "customer_id": 2}
            ]
        ).to_dict()

        inputs = []

        type_map = {
            'FP32': np.float32,
            'INT64': np.int64
        }

        for model_input in self._metadata['inputs']:
            ii = httpclient.InferInput(model_input['name'], shape=[1, 1], datatype=model_input['datatype'])
            ar = np.array(features[model_input['name']], type_map[model_input['datatype']])
            ar = ar.reshape((1, 1))
            ii.set_data_from_numpy(ar)
            inputs.append(ii)

        res = self.triton_client.infer('driver-success-model-onnx', inputs=inputs)

        NumpyCodec.encode_output('probabilities', res.as_numpy('probabilities'))

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                NumpyCodec.encode_output('probabilities', res.as_numpy('probabilities'))
            ]
        )
