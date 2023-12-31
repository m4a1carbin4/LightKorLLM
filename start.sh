#!/usr/bin/env bash

db_image_name="light_kor_llm"
db_container_name="LightKorLLM"
version="0.0.1"

echo "## Automation docker build and run ##"

if [ $# -ne 5 ]; then
 echo "Usage: $0 human_str ai_str producer_topic consumer_topic kafka_server"
 exit -1
else
 echo "param check ok" 
fi

# remove container
echo "=> Remove previous container..."
docker rm -f ${db_container_name}

# remove image
echo "=> Remove previous image..."
docker rmi -f ${db_image_name}:${version}

# new-build/re-build docker image
echo "=> Build new image..."
docker build --build-arg kafka_producer_topic="$3" --build-arg kafka_consumer_topic="$4" --build-arg kafka_server="$5" --build-arg ai_str="$2" --build-arg human_str="$1" --tag ${db_image_name}:${version} .

# Run container
echo "=> Run container..."
docker run -it --net=host --name ${db_container_name} --gpus all ${db_image_name}:${version}