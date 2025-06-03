import torch
import torch_mlu
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

#TODO:设置模型路径,对CodeLlama-7b-hf或CodeLlama-13b-Instruct-hf模型进行加载测试
model_id="/workspace/model/favorite/large-scale-models/model-v1/CodeLlama-7b-hf"
#TODO：利用transformers库函数从预训练模型标识符model_id加载分词器tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_id)
#TODO:利用transformers库函数从预训练模型加载自回归语言模型,配置模型的数据类型为torch.float16、自动选择设备映射，并关闭安全张量选项
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", use_safetensors=False)
#TODO: 将模型设置为评估模式
model = model.eval()
prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
import time
t1 = time.perf_counter()
#TODO:# 使用tokenizer将输入的文本prompt进行分词，并返回PyTorch张量表示的input_ids
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#TODO: 将input_ids张量移动到MLU（Ascend）设备上
input_ids = input_ids.to("mlu")
#TODO： 使用模型生成文本序列，给定输入张量 input_ids，限制最大生成标记数为 240，启用缓存机制
output = model.generate(input_ids, max_new_tokens=240, use_cache=True)
#TODO: 将模型生成的结果从MLU设备移动到CPU设备
output = output.to("cpu")
t2 = time.perf_counter()
#TODO:计算推理延迟
latency = t2 - t1
#TODO:计算输出的token数量
output_token_num = output.size(1)
print('time cost:{} s, output_token_num:{}, total throughout:{} token/s'.\
    format(latency, output_token_num, output_token_num / latency))
#TODO:通过解码生成的张量，获取填充后的文本，跳过特殊标记
filling = tokenizer.decode(output[0], skip_special_tokens=True)
#TODO: 打印替换了占位符"<FILL_ME>"的完整代码
print(filling.replace("<FILL_ME>", "Remove non-ASCII characters fro`m a string and return the cleaned string."))
print("TRANSFORMER CODELLAMA PASS!")
