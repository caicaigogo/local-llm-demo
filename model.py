from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenModel:
    def __init__(self, model_path="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 使用量化加载减少内存占用
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        self.model.eval()

    def generate(self, prompt, max_length=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
