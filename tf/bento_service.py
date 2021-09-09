import bentoml
from typing import List, BinaryIO
from bentoml import api
from redisai_artifact import RedisaiArtifact
from bentoml.adapters import ImageInput,FileInput

import json
from skimage import io
import numpy as np

class_idx = json.load(open("data/imagenet_classes.json"))


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'scikit-image'])
@bentoml.artifacts([RedisaiArtifact('model', 'images', 'output')])
class FashionMnistTensorflow(bentoml.BentoService):

    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        input_datas = []
        for fs in file_streams:
            numpy_img = io.imread(fs).astype(dtype=np.float32)
            numpy_img = np.expand_dims(numpy_img, axis=0) / 255

        input_datas = np.stack(input_datas)
        con = self.artifacts.net
        con.tensorset('images', input_datas, dtype='float')

        # outputs = self.artifacts.net(Variable(torch.stack(input_datas)))
        con.modelrun('model', inputs=['images'], outputs=['outs'])
        outputs = con.tensorget('outs')
        ind = outputs.argmax()
        print(class_idx[str(ind.item() - 1)])