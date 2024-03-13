import json
import numpy as np
import argparse
import re

from sacrebleu.metrics import BLEU, CHRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='pred.jsonl')
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--leveled', action='store_true')
    parser.add_argument('--id_to_difficulty_path', type=str, default='/home/zhangc/minority/prompt_translation_2024/data/sentence_to_level/test_200_id_2_difficulty.json')
    args = parser.parse_args()

    if args.lang == 'zh':
        chrfpp = CHRF(word_order=2, lowercase=True)
        chrf = CHRF(word_order=0, lowercase=True)
        scarebleu = BLEU(lowercase=True, tokenize='zh')
    elif args.lang in ['za', 'eng', 'kgv']:
        chrfpp = CHRF(word_order=2, lowercase=True)
        chrf = CHRF(word_order=0, lowercase=True)
        scarebleu = BLEU(lowercase=True)
    
    # load data
    output = []
    with open(args.output_path, "r") as f:
        for line in f:
            line = json.loads(line)
            output.append(line)

    metrics = {}

    refs = []
    preds = []
    for item in output:
        if args.lang == 'za':
            item['gold'] = item['gold'].replace("’", "'").replace("‘", "'")
            item['pred'] = item['pred'].replace("’", "'").replace("‘", "'")
        refs.append(item['gold'])
        preds.append(item['pred'])


    refs = [refs]
    metrics['sacrebleu'] = scarebleu.corpus_score(preds, refs).score
    metrics['chrf++'] = chrfpp.corpus_score(preds, refs).score
    metrics['chrf'] = chrf.corpus_score(preds, refs).score
    for key in ['sacrebleu', 'chrf++', 'chrf']:
        metrics[key] = np.around(metrics[key], decimals=4)
    print("overall:")
    print(metrics)

    if args.leveled:
        metrics_by_level = {}
        for level in ['easy', 'medium', 'hard']:
            refs = []
            preds = []
            for i, item in enumerate(output):
                if item['source'] == level:
                    if args.lang == 'za':
                        item['gold'] = item['gold'].replace("’", "'").replace("‘", "'")
                        item['pred'] = item['pred'].replace("’", "'").replace("‘", "'")
                    refs.append(item['gold'])
                    preds.append(item['pred'])
            refs = [refs]
            metrics_by_level[level] = {}
            metrics_by_level[level]['sacrebleu'] = scarebleu.corpus_score(preds, refs).score
            metrics_by_level[level]['chrf++'] = chrfpp.corpus_score(preds, refs).score
            metrics_by_level[level]['chrf'] = chrf.corpus_score(preds, refs).score
            for key in ['sacrebleu', 'chrf++', 'chrf']:
                metrics_by_level[level][key] = np.around(metrics_by_level[level][key], decimals=4)
            print(level)
            print(metrics_by_level[level])
        

