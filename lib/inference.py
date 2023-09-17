class inference:

    def __init__(self,model,tokenizer,infer_str):
        self.model = model
        self.tokenizer = tokenizer
        self.infer_str = infer_str

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
        
        return(gened[0])

