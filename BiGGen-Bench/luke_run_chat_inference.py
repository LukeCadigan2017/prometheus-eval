import argparse
import json
import os
import warnings
from pathlib import Path

import huggingface_hub
import pandas as pd
from datasets import load_dataset
from dotenv import dotenv_values

# Run `source init.sh` to use local prometheus_eval
from prometheus_eval.mock import MockLLM
from prometheus_eval.vllm import VLLM
from transformers import AutoTokenizer


def apply_template_hf(tokenizer, record):
    if tokenizer.chat_template is not None and "system" in tokenizer.chat_template:
        messages = [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["input"]},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": record["system_prompt"] + "\n\n" + record["input"],
            }
        ]

    input_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return input_str


def dummy_completions(inputs, **kwargs):
    return ["dummy output"] * len(inputs)


def main(args):
    model_name: str = args.model_name
    output_file_path: str = args.output_file_path
    beam_size:int=args.beam_size
    max_tokens = 2048
    
    toy_example=True
    counter=0
    if(toy_example):
        print("using toy example!!!!")
        max_tokens=128

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dataset: pd.DataFrame = load_dataset(
        "prometheus-eval/BiGGen-Bench", split="test"
    ).to_pandas()

    # records: Full data that has all the information of BiGGen-Bench
    # inputs: Inputs that will be fed to the model
    records = []
    inputs = []

    for row in dataset.iterrows():

        record = row[1]
        records.append(record.to_dict())
        inputs.append(apply_template_hf(tokenizer, record))
        if(toy_example):
            if counter==0:
                print("record is ", row[1])
            if(counter>=0):
                break
            counter+=1
            
   

    params = {
        "max_tokens": max_tokens,
        "use_beam_search":True,
        "repetition_penalty": 1.00,
        "best_of": beam_size,
        "temperature": 1.0,
        "use_tqdm": True,
    }

    # TODO: Support changing and setting the model parameters from the command line
    if toy_example:
        model = VLLM(model_name, dtype='half')
    elif model_name.endswith("AWQ"):
        model = VLLM(model_name, tensor_parallel_size=1, quantization="AWQ")
    elif model_name.endswith("GPTQ"):
        model = VLLM(model_name, tensor_parallel_size=1, quantization="GPTQ")
    else:
        model = VLLM(model_name, tensor_parallel_size=1)


    outputs = model.completions(inputs, **params)

    if len(outputs) != 765:
        warnings.warn(f"Expected 765 outputs, got {len(outputs)}")

    result = {}

    for record, output in zip(records, outputs):
        uid = record["id"]

        result[uid] = record.copy()
        result[uid]["response"] = output.strip()
        result[uid]["response_model_name"] = model_name

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to evaluate. Has to be a valid Hugging Face model name.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
        help="Path to save the output file",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        required=True,
        help="Number of beams",
    )

    hf_token = dotenv_values(".env").get("HF_TOKEN", None)
    if hf_token is not None:
        huggingface_hub.login(token=hf_token)

    args = parser.parse_args()

    main(args)

    
