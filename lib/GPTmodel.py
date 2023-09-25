from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


class AutoGPTQ:

    def __init__(self, base_model_dir, PEFT_model_dir, device):

        self.base_model_dir = base_model_dir
        self.PEFT_model_dir = PEFT_model_dir
        self.device = device
        self.model, self.tokenizer = self.__load_model()

    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_dir, device_map=self.device)

        if self.PEFT_model_dir:

            peft_config = PeftConfig.from_pretrained(self.PEFT_model_dir)

            model = PeftModel.from_pretrained(
                model, self.PEFT_model_dir, config=peft_config)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)

        return model, tokenizer
