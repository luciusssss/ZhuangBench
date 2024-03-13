import json
from tqdm import tqdm

from rank_bm25 import BM25Okapi

from tokenizer import *

lang2tokenizer = {
    'zh': ZhTokenizer(),
    'za': ZaTokenizer(),
    'eng': EngTokenizer(),
    'kgv': KgvTokenizer(),
}

class ParallelCorpus():
    def __init__(self, src_lang, tgt_lang, corpus_path, construct_bm25=True):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.corpus_path = corpus_path
        self.load_corpus()
        if construct_bm25:
            self.construct_bm25()
            self.construct_reversed_index()

    def __len__(self):
        return len(self.corpus) 

    def __getitem__(self, idx):
        return self.corpus[idx]
    
    def load_corpus(self):
        # load corpus
        self.corpus = []
        if self.corpus_path.endswith('.json'):
            self.corpus = json.load(open(self.corpus_path, 'r'))
        elif self.corpus_path.endswith('.jsonl'):
            with open(self.corpus_path, 'r') as f:
                for line in f:
                    json_obj = json.loads(line)
                    src_sentence = json_obj[self.src_lang]
                    tgt_sentence = json_obj[self.tgt_lang]
                    if 'source' in json_obj:
                        source = json_obj['source']
                    else:
                        source = 'n/a'
                    self.corpus.append({
                        self.src_lang: src_sentence,
                        self.tgt_lang: tgt_sentence,
                        'source': source,
                    })
        else:
            raise NotImplementedError("Unsupported corpus format!")
        
    
    def construct_bm25(self):
        self.bm25 = {}
        if self.src_lang == 'zh':
            self.bm25[self.src_lang] = BM25Okapi([lang2tokenizer[self.src_lang].tokenize(item[self.src_lang], remove_punc=True, cut_for_search=True) for item in self.corpus])
        else:
            self.bm25[self.src_lang] = BM25Okapi([lang2tokenizer[self.src_lang].tokenize(item[self.src_lang], remove_punc=True) for item in self.corpus])
        
        if self.tgt_lang == 'zh':
            self.bm25[self.tgt_lang] = BM25Okapi([lang2tokenizer[self.tgt_lang].tokenize(item[self.tgt_lang], remove_punc=True, cut_for_search=True) for item in self.corpus])
        else:
            self.bm25[self.tgt_lang] = BM25Okapi([lang2tokenizer[self.tgt_lang].tokenize(item[self.tgt_lang], remove_punc=True) for item in self.corpus])
    

    def __get_ngram_from_word_list(self, word_list, n):
            ngram_list = []
            for i in range(len(word_list) - n + 1):
                ngram_list.append(' '.join(word_list[i:i+n]))
            # print(ngram_list)
            return ngram_list
    

    def construct_reversed_index(self):
        self.reversed_index = {
            self.src_lang: {},
            self.tgt_lang: {},
        }
        for i, item in enumerate(self.corpus):
            # tokenize src
            if self.src_lang == 'zh':
                src_tokens = lang2tokenizer[self.src_lang].tokenize(item[self.src_lang], remove_punc=True, cut_for_search=True)
            else:
                src_tokens = lang2tokenizer[self.src_lang].tokenize(item[self.src_lang], remove_punc=True)

            for token in src_tokens:
                if token not in self.reversed_index[self.src_lang]:
                    self.reversed_index[self.src_lang][token] = []
                self.reversed_index[self.src_lang][token].append(i)

            # tokenize tgt
            if self.tgt_lang == 'zh':
                tgt_tokens = lang2tokenizer[self.tgt_lang].tokenize(item[self.tgt_lang], remove_punc=True, cut_for_search=True)
            else:
                tgt_tokens = lang2tokenizer[self.tgt_lang].tokenize(item[self.tgt_lang], remove_punc=True)

            for token in tgt_tokens:
                if token not in self.reversed_index[self.tgt_lang]:
                    self.reversed_index[self.tgt_lang][token] = []
                self.reversed_index[self.tgt_lang][token].append(i)

    
    def search_by_bm25(self, text, query_lang='src', top_k=5):
        if query_lang == 'src':
            query_lang = self.src_lang
        elif query_lang == 'tgt':
            query_lang = self.tgt_lang

        
        # tokenize
        if query_lang == 'zh':
            query = lang2tokenizer[query_lang].tokenize(text, remove_punc=True, cut_for_search=True)
        else:
            query = lang2tokenizer[query_lang].tokenize(text, remove_punc=True)
        # print("query:", query)

        # search
        doc_scores = self.bm25[query_lang].get_scores(query)
        top_k_idx = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[-top_k:]
        top_k_sentences_with_scores = [{"pair": self.corpus[i], "score": doc_scores[i]} for i in top_k_idx]
        
        return top_k_sentences_with_scores
    
    def search_for_sentences_by_word(self, word, query_lang='src'):
        word = word.lower().strip()

        if query_lang == 'src':
            query_lang = self.src_lang
        elif query_lang == 'tgt':
            query_lang = self.tgt_lang

        if word not in self.reversed_index[query_lang]:
            return []

        idxs = self.reversed_index[query_lang][word]
        return [self.corpus[i] for i in idxs]
    
    def search_for_sentences_by_word_pair(self, src_word, target_word):
        src_word = src_word.lower().strip()
        target_word = target_word.lower().strip()

        if src_word not in self.reversed_index[self.src_lang] or target_word not in self.reversed_index[self.tgt_lang]:
            return []

        src_idxs = self.reversed_index[self.src_lang][src_word]
        target_idxs = self.reversed_index[self.tgt_lang][target_word]

        intersection = []
        for idx in src_idxs:
            if idx in target_idxs:
                intersection.append(idx)

        return [self.corpus[i] for i in intersection]


    
if __name__ == "__main__":
    pass