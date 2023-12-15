#!/usr/bin/env bash

python main.py --kafka --kafka_producer_topic test_llm --kafka_consumer_topic test_llm_input --quantized_model_dir WGNW/llama-2-7b-ko-auto-gptq-full-v2 --early_stopping --max_new_token 256 --human_str "### 시청자" --ai_str "### 전파녀" --stop_str "### 시청자:"

#python main.py --base_dir ./ --model_name G_67000.pth --config_name Arona.json