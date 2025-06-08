import transformers
import torch
import torch_mlu
import time

model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B/"

# ✅ 创建文本生成的 pipeline，指定任务类型、模型路径、数据类型，并在 MLU 上运行
pipeline = transformers.pipeline(
    task="text-generation",
    model=model_id,
    device=torch.mlu.current_device(),
    torch_dtype=torch.float16
)

messages = [
    {"role": "system", "content": "You are a story writing chatbot"},
    {"role": "user", "content": "Once upon a time, .... start to write a very long story"},
]

# ✅ 应用聊天模板，将消息转化为 prompt（以下假设模型使用 ChatML 或类似格式）
prompt = "<|system|>\nYou are a story writing chatbot\n<|user|>\nOnce upon a time, .... start to write a very long story\n<|assistant|>\n"

# ✅ 终止符号：通常为模型的 `eos_token_id`，加上自定义终止 token ID（如 <|eot_id|>）
# 为了通用，这里取 tokenizer.eos_token_id，如果知道 <|eot_id|> 的 ID，可加入
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
terminators = [tokenizer.eos_token_id]  # 可添加额外自定义 terminator id

times = []
for i in range(1):
    max_length = 256

    # ✅ 记录开始时间
    start_time = time.time()

    # ✅ 执行推理
    outputs = pipeline(
        prompt,
        max_new_tokens=max_length,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # ✅ 记录结束时间
    end_time = time.time()

    # ✅ 计算耗时
    elapsed_time = end_time - start_time

    # ✅ 计算吞吐量（tokens/s）
    tokens_per_sec = max_length / elapsed_time

    # ✅ 存入列表
    times.append(tokens_per_sec)

    print(f"iter: {i}, Tokens per second: {tokens_per_sec:.2f}")

print("========================")
# ✅ 计算平均吞吐量
print("Average tokens per second:", sum(times) / len(times))
print("========================")

print("INFERSPEED PASS!")