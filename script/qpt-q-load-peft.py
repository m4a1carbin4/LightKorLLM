import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GPTQConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from argparse import ArgumentParser

def gen(x,model,tokenizer):

    gened = model.generate(
        **tokenizer(
            f"### 질문: {x}\n\n### 답변:",
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))

def main():
    parser = ArgumentParser()
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--peft_lora_dir", type=str, default=None)

    args = parser.parse_args()

    if not args.quantized_model_dir :
        print("Set the model path or model repo address.")
        return
    else :
        base_model = args.quantized_model_dir

    model = AutoModelForCausalLM.from_pretrained(base_model,device_map="auto")

    if args.peft_lora_dir :

        peft_model = args.peft_lora_dir

        peft_config = PeftConfig.from_pretrained(peft_model)

        model = PeftModel.from_pretrained(model, peft_model,config=peft_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    input_str = ""

    while not input_str == "stop":
        input_str = input("test input : ")

        gen(input_str,model=model,tokenizer=tokenizer)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()