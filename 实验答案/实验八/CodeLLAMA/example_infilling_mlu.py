# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import time
from llama_mlu import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 192,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
):
    #TODO:使用LLAMA模型构建生成器
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        '''def remove_non_ascii(s: str) -> str:
    """ <FILL>
    return result
''',
        """# Installation instructions:
    ```bash
<FILL>
    ```
This downloads the LLaMA inference code and installs the repository as a local pip package.
""",
        """class InterfaceManagerFactory(AbstractManagerFactory):
    def __init__(<FILL>
def main():
    factory = InterfaceManagerFactory(start=datetime.now())
    managers = []
    for i in range(10):
        managers.append(factory.build(id=i))
""",
        """/-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/
theorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :
  π₁ P = 0 ↔ <FILL> = 0 :=
begin
  split,
  { intros h f,
    rw pi_1_etalisation at h,
    simp [h],
    refl
  },
  { intro h,
    have := @quasi_adjoint C D P,
    simp [←pi_1_etalisation, this, h],
    refl
  }
end
""",
    ]
    start_time = time.time()
    prefixes = [p.split("<FILL>")[0] for p in prompts]
    suffixes = [p.split("<FILL>")[1] for p in prompts]
    # 使用LLAMA模型填充文本
    results = generator.text_infilling(
        prefixes=prefixes,
        suffixes=suffixes,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )
    for prompt, result in zip(prompts, results):
        print("\n================= Prompt text =================\n")
        print(prompt)
        print("\n================= Filled text =================\n")
        print(result["full_text"])
    
    print(f"Infer {time.time() - start_time:.6f} seconds")
    print("CODELLAMA INFILLING PASS!")


if __name__ == "__main__":
    fire.Fire(main)
