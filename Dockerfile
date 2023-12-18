FROM wgnwmgm/lightkorllm

RUN ~/.pyenv/bin/pyenv install 3.11

WORKDIR /

RUN git clone https://github.com/m4a1carbin4/LightKorLLM.git

WORKDIR /LightKorLLM

ARG kafka_producer_topic
ARG kafka_consumer_topic
ARG kafka_server="localhost"

ARG ai_str
ARG human_str

ENV ai_str_var=$ai_str
ENV human_str_var=$human_str

ENV consumer_var=$kafka_consumer_topic
ENV producer_var=$kafka_producer_topic
ENV kafka_server_var=$kafka_server

ENTRYPOINT eval "$(~/.pyenv/bin/pyenv init -)" && ~/.pyenv/bin/pyenv local 3.11 && pip install -r requirements.txt && python main.py --kafka --kafka_producer_topic $producer_var --kafka_consumer_topic $consumer_var --quantized_model_dir WGNW/llama-2-7b-ko-auto-gptq-full-v2 --early_stopping --max_new_token 256 --human_str "$human_str_var" --ai_str "$ai_str_var"