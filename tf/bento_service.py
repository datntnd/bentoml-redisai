import bentoml
from typing import List, BinaryIO
from bentoml import api
from redisai_artifact import RedisaiArtifact
from bentoml.adapters import FileInput
import os
import json
from skimage import io
import numpy as np
from skimage.transform import  resize

base_path = os.path.dirname(__file__)
# remember add imagenet_classes.json in folder data/ delploy
class_idx = json.load(open(os.path.join(base_path,"data/imagenet_classes.json")))
 

#luu y: in @artifacts need to covert model to frozen graph (see example/frozen graph tf to detail for tf1 and tf2) 
# and define input note and output note in model
# in call artifacts line 34, name must correspond to initialization (here is 'model').
# type input redisai in float so here use np.float32 not np.float16

@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'scikit-image'])
@bentoml.artifacts([RedisaiArtifact('model', 'tf', 'images', 'output')])
class ImageTFService(bentoml.BentoService):
 
    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        input_datas = []
        for fs in file_streams:
            numpy_img = io.imread(fs).astype(dtype=np.float32)
            numpy_img = np.expand_dims(numpy_img, axis=0) / 255
            numpy_img = resize(numpy_img, (224,224,3),  anti_aliasing=True)
            input_datas.append(numpy_img)
 
        input_datas = np.vstack(input_datas).astype(dtype=np.float32)
        con = self.artifacts.model
        con.tensorset('imgs', input_datas, dtype='float')
 
        con.modelrun('model', 'imgs', 'outs')
        outputs = con.tensorget('outs')
        output_classes = outputs.argmax(axis=1)
        # print([class_idx[str(ind.item() - 1)] for ind in output_classes ])
        return [class_idx[str(ind.item() - 1)] for ind in output_classes ]
