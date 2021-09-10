from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from bento_service import OnnxIrisClassifierService
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


initial_type = [('float_input', FloatTensorType([1, 4]))]
onnx_model = convert_sklearn(clr, initial_types=initial_type)
# with open("rf_iris.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())

svc = OnnxIrisClassifierService()
svc.pack('model', onnx_model)
svc.save()


# Example input: [[5.1, 3.5, 1.4, 0.2]]
# @bentoml.env(infer_pip_packages=True)
# @bentoml.artifacts([OnnxModelArtifact('model', backend='onnxruntime')])
# class OnnxIrisClassifierService(bentoml.BentoService):
#     @bentoml.api(input=DataframeInput(), batch=True)
#     def predict(self, df):
#         input_data = df.to_numpy().astype(numpy.float32
#         input_name = self.artifacts.model.get_inputs()[0].name
#         output_name = self.artifacts.model.get_outputs()[0].name
#         return self.artifacts.model.run(
#                     [output_name], {input_name: input_data}
#                )[0]
# >>>
# svc = OnnxIrisClassifierService()