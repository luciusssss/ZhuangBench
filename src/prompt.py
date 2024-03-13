from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


def construct_prompt_zh2za(src_sent, dictionary, parallel_corpus, args):
    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []


    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 先看是否有精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=2)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"汉语词语“{word}”在壮语中可能的翻译是{concated_meaning}；\n"
            else:
                # 如果没有精确匹配，则看是否有模糊匹配
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=2, max_num_meanings_per_word=2)
                for item in fuzzy_match_meanings[:2]:
                    concated_meaning = "”或“".join(item["meanings"])
                    concated_meaning = "“" + concated_meaning + "”"
                    prompt += f"汉语词语“{item['word']}”在壮语中可能的翻译是{concated_meaning}；\n"
        return prompt
    
    prompt = ""


    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将汉语句子翻译成壮语。\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"## 请将下面的汉语句子翻译成壮语：{item[args.src_lang]}\n"
            
            # 附上参考词汇
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += f"## 所以，该汉语句子完整的壮语翻译是：{item['za']}\n\n"
    
    # prompt最后是需要翻译的句子
    prompt += f"## 请将下面的汉语句子翻译成壮语：{src_sent}\n"
    prompt += get_word_explanation_prompt(src_sent)
    prompt += f"## 所以，该汉语句子完整的壮语翻译是："

    return prompt


def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, args):
    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []


    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 先看是否有精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=2)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"壮语词语“{word}”在汉语中可能的翻译是{concated_meaning}；\n"
            else:
                # 如果没有精确匹配，则看是否有模糊匹配
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=2, max_num_meanings_per_word=2)
                for item in fuzzy_match_meanings[:2]:
                    concated_meaning = "”或“".join(item["meanings"])
                    concated_meaning = "“" + concated_meaning + "”"
                    prompt += f"壮语词语“{item['word']}”在汉语中可能的翻译是{concated_meaning}；\n"
        return prompt
    
    prompt = ""

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将壮语句子翻译成汉语。\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"## 请将下面的壮语句子翻译成汉语：{item[args.src_lang]}\n"

            # 附上参考词汇
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += f"## 所以，该壮语句子完整的汉语翻译是：{item['zh']}\n\n"
    
    # prompt最后是需要翻译的句子
    prompt += f"## 请将下面的壮语句子翻译成汉语：{src_sent}\n"
    prompt += get_word_explanation_prompt(src_sent)
    prompt += f"## 所以，该壮语句子完整的汉语翻译是："

    return prompt








if __name__ == '__main__':
    pass

    



