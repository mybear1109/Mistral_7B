# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
pipe(messages)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


model_id = "mistralai/Mistral-7B-Instruct-v0.3"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)


# 모델 저장 경로
save_dir = "models/mistral_7b_instruct_4bit"

# 모델과 토크나이저 저장
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ 모델 로딩 및 저장 완료: {save_dir}")