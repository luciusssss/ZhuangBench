import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name, model_path, n_gpu=1, use_vllm=True):
    print("loading model...")
    if use_vllm:
        llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=n_gpu, max_model_len=8192)
        print("loaded!")
        return llm
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        print("loaded!")
        return llm, tokenizer
    
    

def get_pred(llm, sampling_params, prompt):
    # print("prompt:", prompt)
    outputs = llm.generate(prompt, sampling_params)
    result = outputs[0].outputs[0].text.strip().split('\n')[0]
    result = result.split("<|endoftext|>")[0].strip()
    result = result.split("<|im_end|>")[0].strip()
    
    # print("result:", result)
    return result

def get_pred_no_vllm(llm, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = len(inputs['input_ids'][0])
    # print("input_len", input_len)
    inputs = inputs.to('cuda')
    preds = llm.generate(
        **inputs,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    pred = tokenizer.decode(preds[0][input_len:], skip_special_tokens=True)
    output = pred
    result = output.strip().split('\n')[0]
    return result


if __name__ == '__main__':
    pass