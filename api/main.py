import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))+"/lib")

from Infer import Infer
from GPTmodel import AutoGPTQ
from controll import apiControll
import json
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GPTQConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from flask_ngrok2 import run_with_ngrok
from flask import Flask, request
from flask_restx import Api, Resource

def main():

    parser = ArgumentParser()
    parser.add_argument("--quantized_model_dir", type=str,
                        default=None, help="main quantized_model_dir")
    parser.add_argument("--peft_lora_dir", type=str,
                        default=None, help="(Optional) custom peft_lora_dir")
    parser.add_argument("--device", type=str, default="cuda",
                        help="(Optional) where to load model (Depending on the model cpu cannot be useable)")
    parser.add_argument("--max_new_token", type=int,
                        default=128, help="for max new token")
    parser.add_argument("--max_history", type=int, default=0,
                        help="how many chat history used (default 0)")
    parser.add_argument("--early_stopping", action="store_true",
                        help="whether use early_stopping")
    parser.add_argument("--port", type=int, default=5989,
                        help="api server port (default 5989)")
    parser.add_argument("--ngrock", action="store_true",
                        help="whether use ngrock")

    args = parser.parse_args()

    main_model = AutoGPTQ(args.quantized_model_dir,
                          args.peft_lora_dir, args.device)

    main_infer = Infer(model=main_model.model, tokenizer=main_model.tokenizer, max_new_tokens=args.max_new_token,
                       early_stopping=args.early_stopping, max_history=args.max_history)

    app = Flask(__name__)

    if args.ngrock:
        run_with_ngrok(app=app, auth_token='1umfu85e4o3OQdknLh3w9ojZXFD_84u1iPX21iTH4Avtzkh9g')
    
    api = Api(app)

    apiControll(api=api,infer=main_infer)

    app.run(debug=True, host='0.0.0.0', port=args.port,use_reloader=False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
