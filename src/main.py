import argparse
import os
import json

import numpy as np

from dictionary import *
from corpus import *
from model import *
from tokenizer import *
from prompt import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # linguistic resources
    parser.add_argument('--src_lang', type=str, default='zh')
    parser.add_argument('--tgt_lang', type=str, default='za')
    parser.add_argument('--dict_path', type=str, default='../data/dictionary_za2zh_web+giza.jsonl.jsonl')
    parser.add_argument('--corpus_path', type=str, default='../data/merged_parallel_sentences_20231225_fixed.json')
    parser.add_argument('--test_data_path', type=str, default='../data/test_200.json')

    # model
    parser.add_argument('--model_name', type=str, default='baichuan2-13b-chat')
    parser.add_argument('--model_path', type=str, default='/newdisk/huggingface_cache/baichuan-inc/Baichuan2-13B-Chat')
    parser.add_argument('--chat_mode', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--no_vllm', action='store_true')

    # cofig for generation
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=100)

    # config for prompt
    parser.add_argument('--prompt_type', type=str, default='za2zh')
    parser.add_argument('--num_parallel_sent', type=int, default=5)

    # output path
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()


    # load dictionary
    dictionary = WordDictionary(args.src_lang, args.tgt_lang, args.dict_path)

    # load corpus
    parallel_corpus = ParallelCorpus(args.src_lang, args.tgt_lang, args.corpus_path)
    if 'pos' in args.prompt_type:
        if args.src_lang == 'zh':
            parallel_corpus.add_pos_information_to_corpus(args.src_lang)
        else:
            parallel_corpus.add_pos_information_to_corpus(args.src_lang, dictionary)

    # load test data
    test_data = json.load(open(args.test_data_path, "r"))

    # load model
    if args.no_vllm:
        llm, tokenizer = load_model(args.model_name, args.model_path, args.n_gpu, use_vllm=False)
    else:
        llm = load_model(args.model_name, args.model_path, args.n_gpu, use_vllm=True)
    
    # sampling params
    if args.do_sample:
        # set seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not args.no_vllm:
        if not args.do_sample:
            sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
        else:
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    # construct prompt 
    prompt_func = None

    prompt_type_to_prompt_func = {
        'za2zh': construct_prompt_za2zh,
        'zh2za': construct_prompt_zh2za,
    }

    if args.prompt_type not in prompt_type_to_prompt_func:
        raise NotImplementedError("Unsupported prompt type!")
    else:
        prompt_func = prompt_type_to_prompt_func[args.prompt_type]


    # chat mode
    if args.chat_mode:
        if 'qwen' in args.model_name and 'chat' in args.model_name:
            chat_template = model_to_chat_template['qwen']


    # output path
    if args.output_path == None:
        args.output_path = f"../output/{args.model_name}_{args.prompt_type}_parallel{args.num_parallel_sent}.jsonl"
    
    while os.path.exists(args.output_path):
        args.output_path = args.output_path + ".new.jsonl"
    print("output_path:", args.output_path)
    fout = open(args.output_path, "w")

    # do test
    for item in tqdm(test_data):
        src_sentence = item[args.src_lang]

        # construct prompt
        prompt = prompt_func(src_sentence, dictionary, parallel_corpus, args)
        # print(prompt)

        # special treatment for chat mode
        if args.chat_mode:
            prompt = chat_template.format(prompt=prompt)

        # generate
        if args.no_vllm:
            pred = get_pred_no_vllm(llm, tokenizer, prompt, args)
        else:
            pred = get_pred(llm, sampling_params, prompt)

        print("input:", src_sentence)
        print("gold:", item[args.tgt_lang])
        print("pred:", pred)

        fout.write(json.dumps({"query": src_sentence, "pred": pred, "gold": item[args.tgt_lang], "prompt": prompt, "source": item['source']}, ensure_ascii=False) + "\n")




    