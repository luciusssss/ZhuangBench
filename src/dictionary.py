import json
from rapidfuzz import process, fuzz
from pprint import pprint

class WordDictionary():
    def __init__(self, src_lang, tgt_lang, dict_path):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dict_path = dict_path
        self.load_dict()
    
    def load_dict(self):
        # load dictionary
        self.word_dict = {}
        self.word_to_source = {}
        with open(self.dict_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                src_word = json_obj[self.src_lang+'_word']
                tgt_meanings = json_obj[self.tgt_lang+'_meanings']
                self.word_dict[src_word] = tgt_meanings
                self.word_to_source[src_word] = json_obj['source']

        # choices for fuzzy matching
        self.choices = list(self.word_dict.keys())
    
    def get_source(self, word):
        if word in self.word_to_source:
            return self.word_to_source[word]
        else:
            return None
    
    def get_meanings_by_exact_match(self, word, max_num_meanings=None):
        if word in self.word_dict:
            if max_num_meanings is None:
                return self.word_dict[word]
            else:
                return self.word_dict[word][:max_num_meanings]
        else:
            return None
        
    
    def get_meanings_by_fuzzy_match(self, word, top_k=10, max_num_meanings_per_word=None):
        # rules for special languages
        if self.src_lang == 'za':
            current_choices = [_ for _ in self.choices if len(_) >= 3]
        else:
            current_choices = self.choices
        
        output = []


        # first: Forward Maximum Matching
        for i in range(len(word), 0, -1):
            if word[:i] in current_choices:
                if max_num_meanings_per_word is None:
                    output.append({
                        "word": word[:i],
                        "meanings": self.word_dict[word[:i]],
                        "score": 100,
                        "source": self.word_to_source[word[:i]]
                    })
                else:
                    output.append({
                        "word": word[:i],
                        "meanings": self.word_dict[word[:i]][:max_num_meanings_per_word],
                        "score": 100,
                        "source": self.word_to_source[word[:i]]
                    })
                break

        # second: Backward Maximum Matching
        for i in range(len(word), 0, -1):
            if word[-i:] in current_choices:
                if word[-i:] in [_["word"] for _ in output]:
                    continue
                if max_num_meanings_per_word is None:
                    output.append({
                        "word": word[-i:],
                        "meanings": self.word_dict[word[-i:]],
                        "score": 100,
                        "source": self.word_to_source[word[-i:]]
                    })
                else:
                    output.append({
                        "word": word[-i:],
                        "meanings": self.word_dict[word[-i:]][:max_num_meanings_per_word],
                        "score": 100,
                        "source": self.word_to_source[word[-i:]]
                    })
                break

        # third: Fuzzy Matching with WRatio
        fuzzy_match_results = process.extract(word, current_choices, scorer=fuzz.WRatio, limit=top_k)
        
        for match in fuzzy_match_results:
            if match[0] in [_["word"] for _ in output]:
                continue

            if max_num_meanings_per_word is None:
                output.append({
                    "word": match[0],
                    "meanings": self.word_dict[match[0]],
                    "score": match[1],
                    "source": self.word_to_source[match[0]]
                })
            else:
                output.append({
                    "word": match[0],
                    "meanings": self.word_dict[match[0]][:max_num_meanings_per_word],
                    "score": match[1],
                    "source": self.word_to_source[match[0]]
                })
        return output


if __name__ == "__main__":
    pass
    