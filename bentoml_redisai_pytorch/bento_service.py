from typing import List, BinaryIO
import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms

import bentoml
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput

from redisai import Client
import ml2rt

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'artifacts/net.pt')
@bentoml.env(pip_packages=['torch', 'numpy', 'torchvision', 'scikit-learn', 'ml2rt', 'redisai'])
@bentoml.artifacts([PytorchModelArtifact('net')])
class PytorchImageClassifier(bentoml.BentoService):
    
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
        con = Client(host='localhost', port=6379, db=0)
        model = ml2rt.load_model(model_path)
        con.modelset('net', 'torch', 'cpu', model)
        con.tensorset('ips', Variable(torch.stack(input_datas)).cpu().detach().numpy(), dtype='float')

        # outputs = self.artifacts.net(Variable(torch.stack(input_datas)))
        con.modelrun('net', inputs=['ips'], outputs=['outs'])
        outputs = con.tensorget('outs')
        output_classes = outputs.argmax(axis=1)

        return [classes[output_class] for output_class in output_classes]   


