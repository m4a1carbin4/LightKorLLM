# LightKorLLM

소규모 프로젝트, 개인 프로젝트를 위한 가벼운 한국어 LLM 프로젝트.

![프로젝트 제작 기념 컨셉이미지 : 소규모 도서관 속 AI와의 대화. (made with stable diffusion)](img/AI_img.png)

프로젝트 제작 기념 컨셉이미지 : 소규모 도서관 속 AI와의 대화. (made with stable diffusion)

## 프로젝트 개요

LightKorLLM 프로젝트는 일반 상용 그래픽 카드 (RTX 30시리즈, GTX 16시리즈 등)에서 사용할 수 있는 LLM 프로젝트를 목표로 만들어졌으며, 소규모 개인 프로젝트 또는 학부 프로젝트 내부에서 활용할 수 있도록 파이선 Flask 서버 형태의 백엔드 서버 또한 제공합니다. 

## AutoGPTQ 양자화 및 PEFT 학습 스크립트

LightKorLLM 프로젝트는 기존 LLaMA, alpaca 계열의 LLM 모델들을 AutoGPTQ 를 통해 양자화하여 사용하는 것을 전제로 하고 있으며, 또한 이렇게 양자화된 LLM 모듈들에 대해 PEFT 방식의 파인튜닝을 위한 스크립트 또한 제공하고 있습니다.

- 해당 스크립트들은 script 경로상에서 확인 가능합니다.
    - script / local
        - AutoGPTQ‎_quantization.py : 기존 LLM 모델에 대한 AutoGPTQ를 활용한 양자화 스크립트.
        - peft_lora_train.py : 양자화된 LLM 모델에 대한 PEFT 방식의 파인튜닝 스크립트.
        - qpt-q-load-peft.py : 모델 테스트를 위한 스크립트.
    - script / colab
        - LightKorLLM_auto_gptq.ipynb : Colab환경을 위한 LLM 모델 양자화 스크립트.
        - LightKorLLM_peft.ipynb : Colab 환경을 위한 PEFT 파인튜닝 스크립트.
        - LightKorLLM_launcher.ipynb : Colab 환경을 위한 Flask 서버 실행 스크립트.

## 사전 양자화 완료 모델 (hugging face)

- [WGNW/llama-2-7b-ko-auto-gptq-full-v2](https://huggingface.co/WGNW/llama-2-7b-ko-auto-gptq-full-v2)
    - 베이스 모델 : [heegyu/llama-2-ko-7b-chat](https://huggingface.co/heegyu/llama-2-ko-7b-chat)
    - 4bit 양자화, 모델 로드시 5GB, inference시 최대 7GB VRAM 사용.
        - max_token 128 설정상에서 테스트 된 내용으로 실제 환경에 따라 차이 존재 가능.
- 이외 추가 모델 양자화 및 테스트를 거쳐 리스트 추가될 예정, 스크립트를 사용하여 직접 원하는 모델 양자화 후 사용 또한 가능.

## Flask API Server

- 양자화된 LLM 모델을 좀더 쉽게 사용하기 위해 작성된 소규모 AI 백엔드 서버 프로젝트.
- 서버 실행 :
    
    ```bash
    python api/main.py --quantized_model_dir WGNW/llama-2-7b-ko-auto-gptq-full-v2 --early_stopping --max_new_token 256 --port 9091 --ngrock
    ```
    
- 서버 상세 실행 옵션 :
    
    ```bash
    usage: main.py [-h] [--quantized_model_dir QUANTIZED_MODEL_DIR] [--peft_lora_dir PEFT_LORA_DIR] [--device DEVICE] [--max_new_token MAX_NEW_TOKEN] [--num_beams NUM_BEAMS] [--max_history MAX_HISTORY] [--early_stopping]
                   [--port PORT] [--ngrock] [--ngrock_token NGROCK_TOKEN] [--human_str HUMAN_STR] [--ai_str AI_STR] [--stop_str STOP_STR]
    
    optional arguments:
      -h, --help            show this help message and exit
      --quantized_model_dir QUANTIZED_MODEL_DIR
                            main quantized_model_dir
      --peft_lora_dir PEFT_LORA_DIR
                            (Optional) custom peft_lora_dir
      --device DEVICE       (Optional) where to load model (Depending on the model cpu cannot be useable)
      --max_new_token MAX_NEW_TOKEN
                            for max new token
      --num_beams NUM_BEAMS
                            for setting beams
      --max_history MAX_HISTORY
                            how many chat history used (default 0)
      --early_stopping      whether use early_stopping
      --port PORT           api server port (default 5989)
      --ngrock              whether use ngrock
      --ngrock_token NGROCK_TOKEN
                            (Optional) ngrock_token
      --human_str HUMAN_STR
                            String to represent user input (ex : ### 사용자)(default)
      --ai_str AI_STR       String to represent AI output (ex : ### AI)(default)
      --stop_str STOP_STR   String to stop generation (ex : ### 사용자)(default)
    ```
    
    - quantized_model_dir : 양자화된 모델 경로 (hugging face 레포 또는 로컬 경로)
    - peft_lora_dir : PEFT 파인튜닝 사용시 해당 파인튜닝된 가중치에 대한 경로 (hugging face 레포 또는 로컬 경로)
    - device : 사용할 device 설정 (pytorch의 device 설정과 동일) : 기본적으로 cuda:0 환경 권장.
    - max_new_token : 최대 생성할 새로운 token 갯수 :
        - VRAM 8GB 이하의 경우 128~256 권장.
    - num_beams : beam 갯수 설정.
    - max_history : 대화형 기능 사용시 이전 대화 기록에 대한 최대 저장 갯수.
    - early_stopping : early_stopping 기능 사용 여부.
    - port : 실행 포트 지정.
    - ngrock : ngrock 을 통한 외부 도매인 공개 설정.
    - ngrock_token : ngrock 토큰 설정 (더 긴 세션 유지 시간)
    - human_str : 사용자 입력 표기용 스트링 지정, # 하단 LLM Inference 확인 요망.
    - ai_str : LLM 결과 표기용 스트링 지정. # 하단 LLM Inference 확인 요망.
    - stop_str : 생성 중단 스트링 지정. (답변 생성 도중 해당 스트링이 생성되는 경우 생성을 중단)

## Flask API Documentation

### base

해당 API 서버는 LLM 모델을 통해 텍스트를 생성하고 생성된 텍스트를 반환하는 일련의 절차를 진행합니다.

### LLM Inference

> 해당 내용에 대한 더 상세한 코드는 lib/Infer.py 내부에서 확인 가능.
> 

모델 Inference의 경우 다음과 같은 스트링 구조의 프롬프트를 통해 진행됩니다.

```json
(instruction 구문 : requestbody.instruction.command)
ex : 당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요.

(human_str : 모델에 따라 적합한 형태로 변환하여 사용 가능 (서버 실행시 설정 가능.))
### 사용자:
(사용자 입력 : requestbody.input)
티라노사우르스보다 쌘 공룡이 있을까?

(ai_str : 모델에 따라 적합한 형태로 변환하여 사용 가능 (서버 실행시 설정 가능.))
### AI:
(AI 결과 출력 요청.)
```

해당 구조를 통해 모델에 inference 하여 그 결과값을 리턴 합니다. 

### API call :

- HTTP POST : http://server/inferweb
    - requst body :
        
        ```json
        {
            "instruction":{
                "command":"당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요."
            },
            "input":"미국 독립전쟁에 대해 레포트 형태로 작성해줘."
        }
        ```
        
    - reponse :
        
        ```json
        {
            "result": " 미국 독립전쟁은 미국의 역사에서 가장 중요한 사건 중 하나로 꼽힙니다. 이 전쟁은 미국 본토에서 영국의 통치가 시작된 지 4세기 만에 진행되었으며, 미국 역사의 시작이라고도 할 수 있습니다.\n\n### 미국 독립전쟁은 대영제국과 미국 본토의 주요 항구인 보스턴에서 발생한 1775년 4월 16일부터 1776년 7월 4일까지 지속되었습니다. 이 전쟁은 독립국가가 되는 주요한 사건 중 하나로 작용하였습니다.\n\n### 미국 독립전쟁은 주로 미국의 권리와 자유를 보장하기 위한 이념과 목적으로 진행되었습니다. 이는 미국 독립선언문에 명시된 자유, 독립, 권리 등 미국의 기본권을 나타내는 요소들로 요약될 수 있습니다.\n\n### 이러한 기본권들은 국민의 자유, 개인의 자유를 보장하고 국가의 주권을 강화하는 목적으로 활용되었습니다. 예를 들어, 미국 독립선언서에서는 다음과 같이 서술하고 있습니다:\n\"우리의 권리의 양도는 우리 자신과 우리 후손들 간의 번영과 행복을 위해 그 의무를 다하라고 말합니다. 하지만 우리는 다른 나라와 달리 우리의 권리를 포기할 수 없습니다. 자유와 권리는 인간의 가장 기본적인 권",
            "history": {
                "history": [
                    {
                        "type": "### 사용자",
                        "str": "미국 독립전쟁에 대해 레포트 형태로 작성해줘."
                    },
                    {
                        "type": "### AI",
                        "str": " 미국 독립전쟁은 미국의 역사에서 가장 중요한 사건 중 하나로 꼽힙니다. 이 전쟁은 미국 본토에서 영국의 통치가 시작된 지 4세기 만에 진행되었으며, 미국 역사의 시작이라고도 할 수 있습니다.\n\n### 미국 독립전쟁은 대영제국과 미국 본토의 주요 항구인 보스턴에서 발생한 1775년 4월 16일부터 1776년 7월 4일까지 지속되었습니다. 이 전쟁은 독립국가가 되는 주요한 사건 중 하나로 작용하였습니다.\n\n### 미국 독립전쟁은 주로 미국의 권리와 자유를 보장하기 위한 이념과 목적으로 진행되었습니다. 이는 미국 독립선언문에 명시된 자유, 독립, 권리 등 미국의 기본권을 나타내는 요소들로 요약될 수 있습니다.\n\n### 이러한 기본권들은 국민의 자유, 개인의 자유를 보장하고 국가의 주권을 강화하는 목적으로 활용되었습니다. 예를 들어, 미국 독립선언서에서는 다음과 같이 서술하고 있습니다:\n\"우리의 권리의 양도는 우리 자신과 우리 후손들 간의 번영과 행복을 위해 그 의무를 다하라고 말합니다. 하지만 우리는 다른 나라와 달리 우리의 권리를 포기할 수 없습니다. 자유와 권리는 인간의 가장 기본적인 권"
                    }
                ]
            }
        }
        ```
        
    - result : inference 결과 생성된 답변.
    - history: 사용자 입력과 해당 입력 결과에 대한 답변.
- HTTP POST : http://server/chatweb
    - request body:
        - 1st request
        
        ```json
        {
            "instruction":{
                "command":"당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요."
            },
            "input":"오늘 기분이 우울한데 음악 리스트 추천해줘.",
            "history": {
                "count": 0,
                "history": [
                ]
            }
        }
        ```
        
        - 2st request
        
        ```json
        {
            "instruction":{
                "command":"당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요."
            },
            "input":"기분이 좀 편해진것 같아.",
            "history": {
                "count": 1,
                "history": [
                    {
                        "type": "### 사용자",
                        "str": "오늘 기분이 우울한데 음악 리스트 추천해줘."
                    },
                    {
                        "type": "### AI",
                        "str": " 우울한 감정을 극복하기 위해 음악은 많은 도움이 됩니다. 몇 가지 추천곡을 드리겠습니다.\n\n1. The Beatles - Hey Jude\n\n이 곡은 The Beatles의 명곡 중 하나입니다. 이 곡은 감정적으로 안정된 상태를 유지하며 우울함을 덜어주는 긍정적인 노래입니다.\n\n2. Adele - Someone Like You\n\nAdele는 그녀의 명곡 중 하나인 Someone Like You로 많은 사랑과 존경을 받고 있습니다. 이 곡은 사랑과 긍정적인 감정을 상기시키고 우울함을 극복하는데 도움이 됩니다.\n\n3. Coldplay - Fragments\n\nColdplay의 Fragments는 슬픔과 우울함을 다룬 감정의 변화와 회복을 그린 노래입니다. 이 곡이 우울한 감정을 치유하는데 도움을 줄 것입니다.\n\n4. Lorraine Walton - We Will Rock Again\n\nLorraine Walton은 미국의 뮤지션으로, 이 곡은 그들의 1999년 데뷔 앨범의 트랙입니다. 이 곡은 어려운 시기를 겪은 후의 회복을 다룬 감동적인 노래입니다.\n\n5. Mariah Carey - Without You\n\nMariah Carey는 세계적인 보컬리스트로 그녀"
                    }
                ]
            }
        }
        ```
        
    - response :
        - 1st chat
        
        ```json
        {
            "result": " 우울한 감정을 극복하기 위해 음악은 많은 도움이 됩니다. 몇 가지 추천곡을 드리겠습니다.\n\n1. The Beatles - Hey Jude\n\n이 곡은 The Beatles의 명곡 중 하나입니다. 이 곡은 감정적으로 안정된 상태를 유지하며 우울함을 덜어주는 긍정적인 노래입니다.\n\n2. Adele - Someone Like You\n\nAdele는 그녀의 명곡 중 하나인 Someone Like You로 많은 사랑과 존경을 받고 있습니다. 이 곡은 사랑과 긍정적인 감정을 상기시키고 우울함을 극복하는데 도움이 됩니다.\n\n3. Coldplay - Fragments\n\nColdplay의 Fragments는 슬픔과 우울함을 다룬 감정의 변화와 회복을 그린 노래입니다. 이 곡이 우울한 감정을 치유하는데 도움을 줄 것입니다.\n\n4. Lorraine Walton - We Will Rock Again\n\nLorraine Walton은 미국의 뮤지션으로, 이 곡은 그들의 1999년 데뷔 앨범의 트랙입니다. 이 곡은 어려운 시기를 겪은 후의 회복을 다룬 감동적인 노래입니다.\n\n5. Mariah Carey - Without You\n\nMariah Carey는 세계적인 보컬리스트로 그녀",
            "history": {
                "count": 1,
                "history": [
                    {
                        "type": "### 사용자",
                        "str": "오늘 기분이 우울한데 음악 리스트 추천해줘."
                    },
                    {
                        "type": "### AI",
                        "str": " 우울한 감정을 극복하기 위해 음악은 많은 도움이 됩니다. 몇 가지 추천곡을 드리겠습니다.\n\n1. The Beatles - Hey Jude\n\n이 곡은 The Beatles의 명곡 중 하나입니다. 이 곡은 감정적으로 안정된 상태를 유지하며 우울함을 덜어주는 긍정적인 노래입니다.\n\n2. Adele - Someone Like You\n\nAdele는 그녀의 명곡 중 하나인 Someone Like You로 많은 사랑과 존경을 받고 있습니다. 이 곡은 사랑과 긍정적인 감정을 상기시키고 우울함을 극복하는데 도움이 됩니다.\n\n3. Coldplay - Fragments\n\nColdplay의 Fragments는 슬픔과 우울함을 다룬 감정의 변화와 회복을 그린 노래입니다. 이 곡이 우울한 감정을 치유하는데 도움을 줄 것입니다.\n\n4. Lorraine Walton - We Will Rock Again\n\nLorraine Walton은 미국의 뮤지션으로, 이 곡은 그들의 1999년 데뷔 앨범의 트랙입니다. 이 곡은 어려운 시기를 겪은 후의 회복을 다룬 감동적인 노래입니다.\n\n5. Mariah Carey - Without You\n\nMariah Carey는 세계적인 보컬리스트로 그녀"
                    }
                ]
            }
        }
        ```
        
        - 2st chat
        
        ```json
        {
            "result": " 그렇군요. 기분이 좋아요.\n\n##",
            "history": {
                "count": 2,
                "history": [
                    {
                        "type": "### 사용자",
                        "str": "오늘 기분이 우울한데 음악 리스트 추천해줘."
                    },
                    {
                        "type": "### AI",
                        "str": " 우울한 감정을 극복하기 위해 음악은 많은 도움이 됩니다. 몇 가지 추천곡을 드리겠습니다.\n\n1. The Beatles - Hey Jude\n\n이 곡은 The Beatles의 명곡 중 하나입니다. 이 곡은 감정적으로 안정된 상태를 유지하며 우울함을 덜어주는 긍정적인 노래입니다.\n\n2. Adele - Someone Like You\n\nAdele는 그녀의 명곡 중 하나인 Someone Like You로 많은 사랑과 존경을 받고 있습니다. 이 곡은 사랑과 긍정적인 감정을 상기시키고 우울함을 극복하는데 도움이 됩니다.\n\n3. Coldplay - Fragments\n\nColdplay의 Fragments는 슬픔과 우울함을 다룬 감정의 변화와 회복을 그린 노래입니다. 이 곡이 우울한 감정을 치유하는데 도움을 줄 것입니다.\n\n4. Lorraine Walton - We Will Rock Again\n\nLorraine Walton은 미국의 뮤지션으로, 이 곡은 그들의 1999년 데뷔 앨범의 트랙입니다. 이 곡은 어려운 시기를 겪은 후의 회복을 다룬 감동적인 노래입니다.\n\n5. Mariah Carey - Without You\n\nMariah Carey는 세계적인 보컬리스트로 그녀"
                    },
                    {
                        "type": "### 사용자",
                        "str": "기분이 좀 편해진것 같아."
                    },
                    {
                        "type": "### AI",
                        "str": " 그렇군요. 기분이 좋아요.\n\n##"
                    }
                ]
            }
        }
        ```
        
    - result : inference 결과 생성된 답변.
    - history: 사용자 입력과 해당 입력에 대한 AI 생성 답변을 순차적으로 기록한 리스트.
