import json
import random
import time
from argparse import ArgumentParser

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer


def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


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
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act,model_file_base_name="model"),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code
    )

    examples = load_data(args.tokenizer_dataset_dir, tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if args.save and args.save_and_upload:
        args.save = None

    if not args.quantized_model_dir:
        args.quantized_model_dir = "./quantized_model"

    if args.save:
        model.save_quantized(args.quantized_model_dir)

    if args.save_and_upload:

        model.save_quantized(args.quantized_model_dir)

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