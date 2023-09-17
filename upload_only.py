import json
import random
import time
from argparse import ArgumentParser

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--tokenizer_dataset_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128, help="group size, -1 means no grouping or full rank")
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument("--num_samples", type=int, default=128, help="how many samples will be used to quantize model")
    parser.add_argument("--save", action="store_true" , help="whether save quantized model to disk")
    parser.add_argument("--save_and_upload", action="store_true", help="whether save quantized model to disk and upload to hugging face")
    parser.add_argument("--upload_to", type=str, default=None, help="hugging face repo dir(uploaded)")

    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument("--use_triton", action="store_true", help="whether use triton to speedup at inference")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="max memory used to load model per gpu")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="max memory used to offload model to cpu")
    parser.add_argument("--quant_batch_size", type=int, default=1, help="examples batch size for quantization")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code when loading model")
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        #use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.quantized_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act,model_file_base_name="pytorch_model.bin"),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code
    )

    if args.save and args.save_and_upload:
        args.save = None

    if not args.quantized_model_dir:
        args.quantized_model_dir = "./quantized_model"

    #if args.save:
        #model.save_quantized(args.quantized_model_dir)

    if args.save_and_upload:

        #model.save_quantized(args.quantized_model_dir)

        if not args.upload_to:
            print("hugging face repo not specific, just save quantized model")
        else :
            try:

                repo_id = args.upload_to
                commit_message = f"AutoGPTQ model for {args.pretrained_model_dir}: {args.bits}bits"
                model.push_to_hub(repo_id, save_dir=args.quantized_model_dir, commit_message=commit_message, use_auth_token=True)
                tokenizer.push_to_hub(repo_id, save_dir=args.quantized_model_dir, commit_message=commit_message, use_auth_token=True)
            except Exception as ex:
                print("The following error occurred during upload. :",ex)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()