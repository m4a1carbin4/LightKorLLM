# -*- coding: utf-8 -*- 

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)+"/api"))
sys.path.append(os.path.join(os.path.dirname(__file__)+"/lib"))

from Infer import Infer
from GPTmodel import AutoGPTQ

from kafka_control import LLM_KafkaControl


from controll import apiControll
from argparse import ArgumentParser
from flask_ngrok2 import run_with_ngrok
from flask import Flask, request
from flask_restx import Api, Resource


def main():

    parser = ArgumentParser()
    parser.add_argument("--flask", action="store_true",
                        help="whether use Flask_api")
    parser.add_argument("--kafka", action="store_true",
                        help="whether use Kafka")
    parser.add_argument("--kafka_producer_topic", type=str,
                        default=None, help="kafka_producer_topic")
    parser.add_argument("--kafka_consumer_topic", type=str,
                        default=None, help="kafka_consumer_topic")
    parser.add_argument("--quantized_model_dir", type=str,
                        default=None, help="main quantized_model_dir")
    parser.add_argument("--peft_lora_dir", type=str,
                        default=None, help="(Optional) custom peft_lora_dir")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="(Optional) where to load model (Depending on the model cpu cannot be useable)")
    parser.add_argument("--max_new_token", type=int,
                        default=128, help="for max new token")
    parser.add_argument("--num_beams", type=int,
                        default=1, help="for setting beams"),
    parser.add_argument("--max_history", type=int, default=0,
                        help="how many chat history used (default 0)")
    parser.add_argument("--early_stopping", action="store_true",
                        help="whether use early_stopping")
    parser.add_argument("--port", type=int, default=5989,
                        help="api server port (default 5989)")
    parser.add_argument("--ngrock", action="store_true",
                        help="whether use ngrock")
    parser.add_argument("--ngrock_token", type=str, default=None,
                        help="(Optional) ngrock_token")
    parser.add_argument("--human_str", type=str, default="### 사용자",
                        help="String to represent user input (ex : ### 사용자)(default)")
    parser.add_argument("--ai_str", type=str, default="### AI",
                        help="String to represent AI output (ex  : ### AI)(default)")
    parser.add_argument("--stop_str", type=str, default="### 선생님:",
                        help="String to stop generation (ex : ### 사용자)(default)")

    args = parser.parse_args()

    main_model = AutoGPTQ(args.quantized_model_dir,
                          args.peft_lora_dir, args.device)

    main_infer = Infer(model=main_model.model, tokenizer=main_model.tokenizer, max_new_tokens=args.max_new_token,early_stopping=args.early_stopping, 
                       max_history=args.max_history, num_beams=args.num_beams,human_str=args.human_str,ai_str=args.ai_str,stop_str=args.stop_str,device=args.device)

    if args.flask :

        app = Flask(__name__)
        
        api = Api(app)

        controll = apiControll(api=api,infer=main_infer)

        if args.ngrock:

            if args.ngrock_token:
                run_with_ngrok(app=app, auth_token=args.ngrock_token)
            else:
                run_with_ngrok(app=app)

        app.run(debug=True, host='0.0.0.0', port=args.port,use_reloader=False)
    elif args.kafka :
        
        if args.kafka_producer_topic == None or args.kafka_consumer_topic == None :
            print("need to set kafka producer, comnsumer topic")
            return 
        else:
            kafka_main = LLM_KafkaControl(main_infer,broker=["localhost:9092", "localhost:9093", "localhost:9094"],topics=[args.kafka_producer_topic,args.kafka_consumer_topic])
            kafka_main.receive_message()
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
