import numpy as np
import bentoml
from bentoml.adapters import DataframeInput
from redisai_artifact import RedisaiArtifact
from bentoml import artifacts


@bentoml.env(pip_dependencies=['onnx', 'numpy','scikit-learn', 'redisai', 'onnxruntime'])
@artifacts([RedisaiArtifact('model', backend='onnx')])
class OnnxIrisClassifierService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        input_data = df.to_numpy().astype(dtype=np.float32)

        con = self.artifacts.model

        input_name = con.modelget('model')['inputs']
        output_name = con.modelget('model')['outputs']
        # print("type", type(input_data))

        # print("input", input_name)
        # # out: ['float_input']
        # print("output", output_name)
        # # out: ['output_label', 'output_probability']
        con.tensorset(input_name[0], input_data, dtype='float')

        con.modelrun("model", inputs=input_name, outputs=output_name)

        outtensor = con.tensorget(output_name[0])

        # print("outtensor", outtensor)

        return outtensor 
