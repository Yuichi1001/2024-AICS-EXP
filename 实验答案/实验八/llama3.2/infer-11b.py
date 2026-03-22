import requests
import torch
import torch_mlu
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-11B-Vision-Instruct/"
#TODO: 加载条件生成模型，指定模型路径、数据类型和MLU设备
model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
#TODO:  从预训练模型加载处理器，使用指定的模型路径
processor = AutoProcessor.from_pretrained(model_id)
#TODO: 打开本地的图像文件
image = Image.open("text2img.jpg")
#TODO: 定义用户消息
messages = [
 {"role": "user", "content": [
 {"type": "image"},
 {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
 ]}
]
#TODO: 应用聊天模板，将消息转化为适合模型的输入格式
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
#TODO: 处理图像和文本输入，将其转换为模型可以接受的格式，并将其转移到 MLU 设备上
inputs = processor(image, input_text, return_tensors="pt").to(model.device)
# 使用模型生成文本，新 token 数量可做限制
output = model.generate(**inputs, max_new_tokens=30)
#TODO:解码模型生成的输出并打印结果，以获取可读的文本形式
print(processor.decode(output[0]))
print("Llama3.2 multimodalchat PASS!")