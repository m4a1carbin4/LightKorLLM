class Infer:

    def __init__(self, model, tokenizer, max_new_tokens, early_stopping, max_history):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_token = max_new_tokens
        self.early_stopping = early_stopping
        self.max_history = max_history

    def __infer_gen(self, x):

        gened = self.model.generate(
            **self.tokenizer(
                x,
                return_tensors='pt',
                return_token_type_ids=False
            ).to('cuda'),
            max_new_tokens=self.max_new_token,
            early_stopping=self.early_stopping,
            do_sample=True,
            eos_token_id=2,
        )

        return (gened[0])

    def __history_appender(self, history: dict):

        current_count = history["count"]

        if current_count < self.max_history:

            history_tmp = history["history"]
        else:
            history_tmp = history["history"][-(self.max_history-2):]

        infer_str = ""

        for obj in history_tmp:

            infer_str += f"\n\n {obj['type']}: {obj['str']}"

        return infer_str

    def __history_adder(self, input_str, gen_str, history):

        history_tmp = history["history"]

        history_tmp.append({'type': "human", 'str': input_str})
        history_tmp.append({'type': "Assistent", 'str': gen_str})

        history["history"] = history_tmp

        history["count"] += 1

        return history

    def text_gen(self, input_str, history: dict):

        infer_str = self.__history_appender(history=history)

        infer_str += f"\n\n human: {input_str} \n\n Assistent: "

        gen_token = self.__infer_gen(infer_str)

        gen_str = self.tokenizer.decode(gen_token)

        print(gen_str)

        gen_str = gen_str[len(infer_str):]

        history = self.__history_adder(input_str, gen_str, history)

        return gen_str, history