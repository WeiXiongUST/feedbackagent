#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    #stop=["<|user|>"],
)

def generate_prompt(problem, reasoning_path):
    """
    Generates a prompt for obtaining feedback on a reasoning path, explicitly instructing not to solve the problem.

    Parameters:
        problem (str): The mathematical problem statement.
        reasoning_path (str): The step-by-step reasoning provided for solving the problem.

    Returns:
        str: A formatted prompt to elicit detailed feedback from a language model.
    """
    template = f"""
You are a mathematical reasoning assistant. Do not attempt to solve the problem yourself. Instead, focus solely on analyzing the provided reasoning path. Your task is to:  
1. Identify errors, inconsistencies, or incomplete reasoning in the provided steps.  
2. Suggest corrections or enhancements to improve clarity, rigor, and correctness.  

---

**Problem**:  
{problem}

---

**Reasoning Path**:  
{reasoning_path}

---

**Your Feedback**:  
1. **Reflection**: Highlight specific errors, inconsistencies, or unclear reasoning. Do not solve the problem but explain why the identified issues are problematic.  
2. **Improvement Plan**: Propose corrections or suggest clearer, more rigorous explanations for the identified issues.  

Focus on identifying errors and improving the reasoning process rather than providing the solution.
"""
    return template
  
ds = load_dataset(script_args.dataset_name_or_path, split="train")

def get_prompt(example):
    #tmp = [generate_prompt(example['problem'], resp) for resp in example['responses']]
    tmp = generate_prompt(example['problem'], example['response'])
    return {"prompt": tokenizer.apply_chat_template([{"role":"user", "content":tmp}], tokenize=False, add_generation_prompt=True)}
    
ds = ds.map(get_prompt)

data_size = len(ds["prompt"])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
print(ds, script_args.dataset_name_or_path)
print(ds[0])


prompts = ds["prompt"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"prompt": ds[i]['prompt']], "responses": [out.text for out in output.outputs], "problem": ds[i]['prompt'], "response": ds[i]['response']}
    gathered_data.append(tmp_data)


print("I collect ", len(gathered_data), "samples")


with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')
