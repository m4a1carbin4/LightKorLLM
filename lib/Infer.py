# -*- coding: utf-8 -*- 

from transformers import StoppingCriteria, StoppingCriteriaList
import torch

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids:torch.LongTensor, scores:torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class Infer:

    def __init__(self, model, tokenizer, max_new_tokens, early_stopping, max_history, num_beams, human_str, ai_str, stop_str, device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_token = max_new_tokens
        self.early_stopping = early_stopping
        self.max_history = max_history
        self.num_beams = num_beams
        self.human_str = human_str
        self.ai_str = ai_str
        self.device = device


        print(stop_str)
        stop_words = [stop_str,"###","선생님:","호시노:"]

        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').to(self.device).squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def infer_gen(self, x):

        input_token = self.tokenizer(
                x,
                return_tensors='pt',
                return_token_type_ids=False
            ).to(self.device)

        gened = self.model.generate(
            **input_token,
            max_new_tokens=self.max_new_token,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
            stopping_criteria=self.stopping_criteria,
            do_sample=True,
            eos_token_id=2,
        )

        input_length = 1 if self.model.config.is_encoder_decoder else input_token.input_ids.shape[1]
        gened = gened[:, input_length:]

        return (gened[0])

    def __history_appender(self, history: dict, instruction: dict):

        if instruction :
            instruction_str = instruction["command"] + "\n"
        else:
            instruction_str = ""

        current_count = history["count"]

        if current_count < self.max_history:

            history_tmp = history["history"]
        else:
            history_tmp = history["history"][-(self.max_history-2):]

        infer_str = ""

        infer_str = instruction_str + infer_str

        for obj in history_tmp:

            infer_str += f"\n\n {obj['type']}: {obj['str']}"

        return infer_str
    
    def __instruction_appender(self, instruction: dict):

        
        instruction_str = instruction["command"] + "\n"

        infer_str = ""

        infer_str = instruction_str + infer_str

        return infer_str

    def __history_adder(self, input_str, gen_str, history):

        history_tmp = history["history"]

        history_tmp.append({'type': self.human_str, 'str': input_str})
        history_tmp.append({'type': self.ai_str, 'str': gen_str})

        history["history"] = history_tmp

        history["count"] += 1

        return history
    
    def __infer_return(self, input_str, gen_str):
        
        result_tmp = []

        result_tmp.append({'type': self.human_str, 'str': input_str})
        result_tmp.append({'type': self.ai_str, 'str': gen_str})

        history = {"history":result_tmp}

        return history

    def text_gen(self, data:dict,type:str):

        if type == "chat":
            infer_str = self.__history_appender(history=data["history"],instruction=data["instruction"])
        elif type == "infer":
            infer_str = self.__instruction_appender(instruction=data["instruction"])

        infer_str += f"\n\n{self.human_str}: {data['input']}\n\n{self.ai_str}:"

        gen_token = self.infer_gen(infer_str)

        gen_str = self.tokenizer.decode(gen_token)

        print(gen_str)

        gen_str = gen_str.replace("</s>",'')

        if type == "infer":

            history = self.__infer_return(data['input'], gen_str)

            return gen_str, history
        
        else :

            history = self.__history_adder(data['input'], gen_str, data["history"])

            return gen_str, history
