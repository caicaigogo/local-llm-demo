import unittest
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class TestTransformer(unittest.TestCase):

    def setUp(self):
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
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

    def test_tokenizer(self):
        prompt = '你好'

        # {'input_ids': tensor([[108386]]), 'attention_mask': tensor([[1]])}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def test_generate(self):
        prompt = '你好'

        # {'input_ids': tensor([[108386]]), 'attention_mask': tensor([[1]])}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs
            )
            # tensor([[108386, 3837, 35946, 104044, 100681, 101099, 101895, 110237, 3837,
            #          99172, 100703, 100158, 103998, 1773, 6567, 60757, 52801, 6313,
            #          112169, 87026, 50404]])
            print(outputs)

        # 你好，我最近感觉身体有些不舒服，想咨询一下医生。 您好！很高兴您选择
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)


    def test_stream_generate(self):
        # 创建 streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 输入文本
        prompt = "Once upon a time,"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 在新线程中运行 generate
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 主线程：逐段打印生成的文本
        print(prompt, end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)