- Run docker redisai if cpu:
```
docker run -dp 6379:6379 -it --rm  redislabs/redisai:edge-cpu-bionic
```
and if gpu:
```
docker run -p 6379:6379 --gpus all -it --rm redislabs/redisai:edge-gpu-bionic
```

See all tag image rediasai [here](https://hub.docker.com/r/redislabs/redisai/tags?page=1&ordering=last_updated)

- pip install 
```
pip install -r requirement.txt
```

- Train model:
```
python train.py
```

- Run serve bentoml
```
bentoml serve OnnxIrisClassifierService:latest --enable-microbatch
```