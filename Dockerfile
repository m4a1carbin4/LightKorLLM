FROM wgnwmgm/lightkorllm

RUN ~/.pyenv/bin/pyenv install 3.11

WORKDIR /

RUN git clone https://github.com/m4a1carbin4/LightKorLLM.git

WORKDIR /LightKorLLM

ARG ai_str
ARG human_str
ARG stop_str

ENV ai_str_var=$ai_str
ENV human_str_var=$human_str
ENV stop_str_var=$stop_str

ENTRYPOINT eval "$(~/.pyenv/bin/pyenv init -)" && ~/.pyenv/bin/pyenv local 3.11 && pip install -r requirements.txt && python main.py --kafka --kafka_producer_topic test_llm --kafka_consumer_topic test_llm_input --quantized_model_dir WGNW/llama-2-7b-ko-auto-gptq-full-v2 --early_stopping --max_new_token 256 --human_str "$human_str_var" --ai_str "$ai_str_var" --stop_str "$stop_str_var"