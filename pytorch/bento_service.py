# my_bento_service.py

from redisai_artifact import RedisaiArtifact
from bentoml import BentoService, env, api, artifacts
from bentoml.adapters import JsonInput,FileInput
import bentoml

from torchvision import transforms
import torch
from torch.autograd import Variable

from typing import List, BinaryIO
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

@bentoml.env(pip_packages=['torch', 'numpy', 'torchvision', 'scikit-learn', 'redisai'])
@artifacts([RedisaiArtifact('net')])
class PytorchService(bentoml.BentoService):

    @bentoml.utils.cached_property
    def transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        input_datas = []
        for fs in file_streams:
            img = Image.open(fs).resize((32, 32))
            input_datas.append(self.transform(img))

        con = self.artifacts.net
        con.tensorset('ips', Variable(torch.stack(input_datas)).cpu().detach().numpy(), dtype='float')

        # outputs = self.artifacts.net(Variable(torch.stack(input_datas)))
        con.modelrun('net', inputs=['ips'], outputs=['outs'])
        outputs = con.tensorget('outs')
        output_classes = outputs.argmax(axis=1)

        return [classes[output_class] for output_class in output_classes]   