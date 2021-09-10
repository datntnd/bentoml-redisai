# Example for tf (detail is tf v1).
- Run docker redisai if cpu:
```
docker run -dp 6379:6379 -it --rm  redislabs/redisai:edge-cpu-bionic
```
and if gpu:
```
docker run -dp 6379:6379 --gpus all -it --rm redislabs/redisai:edge-gpu-bionic
```

See all tag image rediasai [here](https://hub.docker.com/r/redislabs/redisai/tags?page=1&ordering=last_updated)

- pip install:
```
pip install -r requirement.txt
```

- Train model and pack to bentoml:
```
python train.py
```

- Copy file *tf/data/imagenet_classes.json* to folder deloy, 
  + Make a folder data in deloy folder, exmaple: /home/user_name/bentoml/repository/ImageTFService/20210910115026_4D8C88/ImageTFService
 
  + Paste file  imagenet_classes.json

- Run bentoml serve
```
bentoml serve ImageTFService:latest --enable-microbatch
```

# Warning:
-  Python 3.8 support requires TensorFlow 2.2 or later and dont cant use tf v1
# Referent:

[imagenet](https://github.com/RedisAI/redisai-examples/tree/master/models/tensorflow/imagenet)

[Frozen-Graph-TensorFlow](https://github.com/leimao/Frozen-Graph-TensorFlow)

Conver to frozen graph tensorflow:

- [tf1](https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/)

- [tf2](https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/)