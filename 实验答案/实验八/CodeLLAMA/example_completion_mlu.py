# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import time
from llama_mlu import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    #TODO: 使用LLAMA模型构建生成器
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """\
def fizzbuzz(n: int):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"""
    ]
    start_time = time.time()

    #TODO: 使用LLAMA模型生成文本
    results = generator.text_completion(
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
    
    print(f"Infer {time.time() - start_time:.6f} seconds")
    print("CODELLAMA COMPLETION PASS!")

if __name__ == "__main__":
    fire.Fire(main)

