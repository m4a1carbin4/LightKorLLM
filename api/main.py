import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))+"/lib")

from flask_restx import Api, Resource
from flask import Flask, request
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GPTQConfig, BitsAndBytesConfig
import torch
from argparse import ArgumentParser
import json
import random
import time
from GPTmodel import AutoGPTQ
from Infer import Infer

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

    args = parser.parse_args()

    main_model = AutoGPTQ(args.quantized_model_dir,
                          args.peft_lora_dir, args.device)

    main_infer = Infer(model=main_model.model, tokenizer=main_model.tokenizer, max_new_tokens=args.max_new_token,
                       early_stopping=args.early_stopping, max_history=args.max_history)

    app = Flask(__name__)
    api = Api(app)

    @api.route('/inferWeb')
    class inferweb(Resource):

        def post(self):

            input_str = request.json.get("inputString")

            history = request.json.get("history")

            result_str, result_history = main_infer.text_gen(
                input_str=input_str, history=history)

            return {
                "result": result_str,
                "history": result_history
            }

    app.run(debug=True, host='0.0.0.0', port=args.port)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
