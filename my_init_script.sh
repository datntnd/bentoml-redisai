#!/usr/bin/env bash

if ! [ -x "$(command -v docker)" ]; then
    echo "You are missing Docker"
    echo "docker not found!"
    echo "Get it here : https://docs.docker.com/engine/install/"
    exit 1
else
    docker -v
fi
if ! [ -x "$(command -v docker-compose)" ]; then
    echo "You are missing Docker Compose"
    echo "docker-compose not found!"
    echo "Get it here : https://docs.docker.com/compose/install/"
    exit 1
else
    docker-compose -v
fi

name='redislabs'
if [ ! "$(docker ps -q -f name=$name)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$name)" ]; then
        # cleanup
        docker rm $name 
    fi
    # run your container
    docker run -dp 6379:6379 -it --rm  redislabs/redisai:edge-cpu-bionic
fi
