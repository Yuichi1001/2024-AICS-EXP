import requests
import torch
import torch_mlu
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-11B-Vision-Instruct/"

# ✅ 加载条件生成模型，指定模型路径、数据类型和MLU设备
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("mlu")

# ✅ 从预训练模型加载处理器
processor = AutoProcessor.from_pretrained(model_id)

# ✅ 打开本地图像文件
image = Image.open("text2img.jpg")  # <-- 请替换为实际图像路径

# ✅ 定义用户消息，包含图片和文本
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]

# ✅ 应用聊天模板（如支持 ChatML 格式）
input_text = processor.apply_chat_template(messages, tokenize=False)

# ✅ 处理图像和文本输入
inputs = processor(
    text=input_text,
    images=image,
    return_tensors="pt"
).to("mlu")

# ✅ 使用模型生成文本
output = model.generate(**inputs, max_new_tokens=64)

# ✅ 解码输出
generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

print("Llama3.2 multimodalchat PASS!")