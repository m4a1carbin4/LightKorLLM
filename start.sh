#!/usr/bin/env bash

cd LightKorLLM

pyenv virtualenv 3.11 LightKorLLM

pyenv activate LightKorLLM

pip install -r requirements.txt

python main.py --kafka --kafka_producer_topic test_llm --kafka_consumer_topic test_llm_input --quantized_model_dir WGNW/llama-2-7b-ko-auto-gptq-full-v2 --early_stopping --max_new_token 256 --human_str $1 --ai_str $2 --stop_str $3