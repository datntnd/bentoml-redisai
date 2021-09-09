# my_model_artifact.py

import os
import json
from bentoml.utils import cloudpickle
from bentoml.exceptions import InvalidArgument
from bentoml.service.artifacts import BentoServiceArtifact
from redisai import Client
import ml2rt
import config
import subprocess
import time
import logging
import redis
import torch

logger = logging.getLogger(__name__)

class Config(dict):
    def __init__(self):
        super().__init__()
        self['host'] = config.REDIS_HOST
        self['port'] = config.REDIS_PORT
        self['username'] = config.REDIS_USERNAME
        self['password'] = config.REDIS_PASSWORD
        self['db'] = config.REDIS_DB

class MyModelArtifact(BentoServiceArtifact):
    def __init__(self, name, file_extension=".pt"):
        super(MyModelArtifact, self).__init__(name)
        self.con = Client(host='localhost', port=6379, db=0)
        self._file_extension = file_extension
        self._model = None

    def pack(self, model):
        self._model = model
        
        return self

    def get(self):
        return self.con

    def save(self, dst, backend='torch', device='cpu'):
        torch.jit.save(self._model, self._file_path(dst))
        model =  ml2rt.load_model(self._file_path(dst))
        self.con.modelset(self.name, backend, device, model)

        return self

    def load(self, path):
        with open(self._file_path(path), 'rb') as f:
            model = f.read()
        # model = ml2rt.load_model(self._file_path(path))
        return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)



