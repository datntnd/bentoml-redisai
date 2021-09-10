# my_model_artifact.py
import os
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    MissingDependencyException,
)

from bentoml.service.artifacts import BentoServiceArtifact
# from redisai import Client
import redisai as rai


class Config(dict):
    def __init__(self):
        super().__init__()
        self['host'] = 'localhost'
        self['port'] = 6379
        self['username'] = None
        self['password'] = None
        self['db'] = 0

SUPPORTED_BACKEND = ["torch", "tf", "onnx"]

class RedisaiArtifact(BentoServiceArtifact):
    def __init__(self, name, backend='torch', input = None, output=None):
        super(RedisaiArtifact, self).__init__(name)
        if backend not in SUPPORTED_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for RedisaiArtifact'
            )
        server_config = Config()
        self.con = rai.Client(**server_config)
        self._input=input
        self._output=output
        self._backend= backend
        self._device = None
        self._model = None

        if self._backend == 'torch':
            self._file_extension = '.pt'
            try:
                import torch
            except ImportError:
                raise MissingDependencyException(
                    "torch package is required to use RedisaiArtifact"
                )

        elif self._backend == 'tf':
            self._file_extension = '.pb'
            try:
                import tensorflow as tf
            except ImportError:
                raise MissingDependencyException(
                    "Tensorflow package is required to use RedisaiArtifact."
                )
        elif self._backend == 'onnx':
            self._file_extension = '.onnx'
            try:
                import onnx

            except ImportError:
                raise InvalidArgument(
                    "ONNX package is required to use RedisaiArtifact."
                )
            
        

    def pack(self, model, device='cpu', input=None):
        # print("model", model)
        self._device = device
        self._model = model
        self._input  = input
        
        return self

    def get(self):
        return self.con

    def save(self, dst):
        if self._backend == 'torch':
            import torch
            torch.jit.save(self._model, self._file_path(dst))
        elif self._backend == 'tf':
            import tensorflow as tf

            TF2 = tf.__version__.startswith('2')
            if TF2:
                tf.io.write_graph(graph_or_graph_def=self._model.graph,
                  logdir=dst,
                  name= self.name + self._file_extension,
                  as_text=False)
            else:
                tf.compat.v1.disable_eager_execution()
                tf.train.write_graph(self._model, dst, self.name + self._file_extension, as_text=False)
        elif self._backend == 'onnx':
            print('OK')
            import onnx
            onnx.save_model(self._model, self._file_path(dst))

            
        # model =  ml2rt.load_model(self._file_path(dst))
        with open(self._file_path(dst), 'rb') as f:
            model = f.read()
        self.con.modelstore(self.name, self._backend, self._device, model, inputs=self._input, outputs=self._output)

        return self

    def load(self, path):
        with open(self._file_path(path), 'rb') as f:
            model = f.read()
        # model = ml2rt.load_model(self._file_path(path))
        return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)



