FROM wgnwmgm/lightkorllm

ARG ai_str="### 호시노"

ARG human_str="### 선생님"

ARG stop_str="### 선생님:"

RUN git clone https://github.com/m4a1carbin4/LightKorLLM.git

RUN cd LightKorLLM

RUN chmod +x start.sh

RUN ./start.sh ${human_str} ${ai_str} ${stop_str}