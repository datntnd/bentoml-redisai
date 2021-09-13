import os
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    MissingDependencyException,
)

from bentoml.service.artifacts import BentoServiceArtifact
import redisai as rai
import logging

logger = logging.getLogger(__name__)


class Config(dict):
    def __init__(self):
        super().__init__()
        self['host'] = 'localhost'
        self['port'] = 6379
        self['username'] = None
        self['password'] = None
        self['db'] = 0


SUPPORTED_BACKEND = ["torch", "tf", "onnx"]
file_extension = {
    'torch': '.pt',
    'tf': '.pb',
    'onnx': '.onnx'}


class RedisaiArtifact(BentoServiceArtifact):
    def __init__(self, name, backend='torch', input=None, output=None, device='cpu'):
        super(RedisaiArtifact, self).__init__(name)
        if backend not in SUPPORTED_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for RedisaiArtifact'
            )
        server_config = Config()
        self.con = rai.Client(**server_config)
        self._input = input
        self._output = output
        self._backend = backend
        self._device = device
        self._model = None

    def pack(self, model):
        if self._backend == 'torch':
            try:
                import torch
            except ImportError:
                raise MissingDependencyException(
                    "torch package is required to use RedisaiArtifact"
                )

            if not isinstance(model, torch.nn.Module):
                raise InvalidArgument(
                    "RedisaiArtifact can only pack type 'torch.jit.ScriptModule'"
                )

        elif self._backend == 'tf':
            try:
                import tensorflow as tf

            except ImportError:
                raise MissingDependencyException(
                    "Tensorflow package is required to use RedisaiArtifact."
                )

        elif self._backend == 'onnx':
            try:
                import onnx

                if not isinstance(model, onnx.ModelProto):
                    raise InvalidArgument(
                        "onnx.ModelProto model file path is required to "
                        "pack an RedisaiArtifact"
                    )

            except ImportError:
                raise InvalidArgument(
                    "ONNX package is required to use RedisaiArtifact."
                )

        self._model = model

        return self

    def get(self):
        return self.con

    def save(self, dst):

        if self._backend == 'torch':
            import torch

            if self._model.training is True:
                logger.warn('Graph is in training mode. Converting to evaluation mode')

                self._model.eval()

            torch.jit.save(self._model, self._file_path(dst))

        elif self._backend == 'tf':
            import tensorflow as tf

            TF2 = tf.__version__.startswith('2')
            if TF2:
                tf.io.write_graph(graph_or_graph_def=self._model.graph,
                                  logdir=dst,
                                  name=self.name + file_extension[self._backend],
                                  as_text=False)
            else:
                tf.compat.v1.disable_eager_execution()
                tf.train.write_graph(self._model, dst, self.name + file_extension[self._backend], as_text=False)

        elif self._backend == 'onnx':
            import onnx
            onnx.save_model(self._model, self._file_path(dst))

        # model =  ml2rt.load_model(self._file_path(dst))
        # with open(self._file_path(dst), 'rb') as f:
        #     model_redisai = f.read()
        # self.con.modelstore(self.name, self._backend, self._device, model_redisai, inputs=self._input,
        #                     outputs=self._output)

        return self

    def load(self, path):
        with open(self._file_path(path), 'rb') as f:
            model_redisai = f.read()

        self.con.modelstore(self.name, self._backend, self._device, model_redisai, inputs=self._input,
                            outputs=self._output)

        # model = ml2rt.load_model(self._file_path(path))
        # return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + file_extension[self._backend])
