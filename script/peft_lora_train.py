from datasets import load_dataset
from functools import reduce
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from argparse import ArgumentParser

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model_dir",type=str, default=None)
    parser.add_argument("--dataset_dir",type=str, default=None)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_step",type=int,default=4)
    parser.add_argument("--train_warmup_steps",type=int, default=2)
    parser.add_argument("--max_train_step",type=int,default=10)
    parser.add_argument("--learning_rate",type=float,default=2e-4)
    parser.add_argument("--train_output_dir",type=str,default="outputs")
    parser.add_argument("--train_optimizer",type=str,default="paged_adamw_8bit")
    parser.add_argument("--upload_dir",type=str,default=None)


    args = parser.parse_args()

    data = load_dataset("junelee/remon_without_nsfw")

    data = data.map(
        lambda x:{
            'text':reduce(lambda acc,cur: acc + (f"##Human:{cur['value']} \n\n"if cur['from'] == 'human' else f"##Assistant: {cur['value']} \n\n"),x['conversations'],"")
        }
    ) # 데이터셋 평문화.

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_dir,device_map="auto")

    data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["k_proj","o_proj","q_proj","v_proj"],
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_step,
            warmup_steps=args.train_warmup_steps,
            max_steps=args.max_train_step,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir=args.train_output_dir,
            optim=args.train_optimizer
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    try:
        repo_id = args.upload_dir
        commit_message = f"AutoGPTQ model for {repo_id}: {repo_id}bits"
        model.push_to_hub(repo_id, save_dir=repo_id, commit_message=commit_message, use_auth_token=True)
        tokenizer.push_to_hub(repo_id, save_dir=repo_id, commit_message=commit_message, use_auth_token=True)

    except Exception as ex:
        print("The following error occurred during upload. :",ex)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()