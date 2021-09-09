# my_model_artifact.py
import os
import json
from bentoml.exceptions import (
    InvalidArgument,
    MissingDependencyException,
)

from bentoml.service.artifacts import BentoServiceArtifact
from redisai import Client
import config

# SUPPORTED_DEPLO_FLAVORS = ['pytorch', 'tensorflow']
# flavor2backend = {
#     'pytorch': 'torch',
#     'tensorflow': 'tf'}

class Config(dict):
    def __init__(self):
        super().__init__()
        self['host'] = config.REDIS_HOST
        self['port'] = config.REDIS_PORT
        self['username'] = config.REDIS_USERNAME
        self['password'] = config.REDIS_PASSWORD
        self['db'] = config.REDIS_DB

class RedisaiArtifact(BentoServiceArtifact):
    def __init__(self, name, input = None, output=None):
        super(RedisaiArtifact, self).__init__(name)
        self.con = Client(host='localhost', port=6379, db=0)
        self._input=input
        self._output=output
        self._model = None
        self._backend= None
        self._file_extension = None

    def pack(self, model,  backend='torch', device='cpu'):
        print("backend", backend)
        self._backend = backend
        self._device = device

        if self._backend == 'torch':
            try:
                import torch
            except ImportError:
                raise MissingDependencyException(
                    "torch package is required to use RedisaiArtifact"
                )

            if not isinstance(model, torch.nn.Module):
                raise InvalidArgument(
                    "RedisaiArtifact can only pack type \
                    'torch.nn.Module' or 'torch.jit.ScriptModule'"
                )
        elif self._backend == 'tf':
            try:
                import tensorflow as tf

                
            except ImportError:
                raise MissingDependencyException(
                    "Tensorflow package is required to use RedisaiArtifact."
                )

        self._model = model
        
        return self

    def get(self):
        return self.con

    def save(self, dst):
        if self._backend == 'torch':
            import torch

            self._file_extension = '.pt'
            torch.jit.save(self._model, self._file_path(dst))
        elif self._backend == 'tf':
            import tensorflow as tf
            self._file_extension = '.pt'

            TF2 = tf.__version__.startswith('2')
            # test with tf 1
            TF2 = False
            if TF2:
                tf.io.write_graph(graph_or_graph_def=self._model.graph,
                  logdir=dst,
                  name= self.name + self._file_extension,
                  as_text=False)
            else:
                import tensorflow.compat.v1 as tf
                tf.compat.v1.disable_eager_execution()
                tf.train.write_graph(self._model, dst, self.name + self._file_extension, as_text=False)

            
        # model =  ml2rt.load_model(self._file_path(dst))
        with open(self._file_path(dst), 'rb') as f:
            model = f.read()
        self.con.modelset(self.name, self._backend, self._device, model, inputs=self._input, outputs=self._output)

        return self

    def load(self, path):
        with open(self._file_path(path), 'rb') as f:
            model = f.read()
        # model = ml2rt.load_model(self._file_path(path))
        return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)



