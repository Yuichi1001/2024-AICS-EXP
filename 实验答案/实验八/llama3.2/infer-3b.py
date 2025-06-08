import torch
import torch_mlu
from transformers import pipeline  # ✅ 导入 pipeline

model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B/"

# ✅ 创建文本生成管道，指定模型路径、数据类型、设备为 MLU
pipe = pipeline(
    task="text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device="mlu"
)

# ✅ 传入提示词进行生成
output = pipe("The key to life is")[0]

# ✅ 提取并打印生成的文本
print(output['generated_text'])
print("Llama3.2 textchat PASS!")
