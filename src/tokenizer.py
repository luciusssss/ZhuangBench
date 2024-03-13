import re
import jieba
import jieba.posseg as pseg

class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # 替换单引号
        text = text.replace("‘", "'").replace("’", "'")
        # 按空格分词
        tokenized_text = text.split(" ")
        # 去除空格
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text

class EngTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()


class KgvTokenizer(Tokenizer):
    def __init__(self):
        super().__init__() 

class ZaTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # 替换单引号
        text = text.replace("‘", "'").replace("’", "'")
        # 按空格分词
        tokenized_text = text.split(" ")
        # 去除空格
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text


class ZhTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False, do_cut_all=False, cut_for_search=False):
        # 使用jieba分词
        text = text.lower()
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, "")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        
        if cut_for_search:
            tokenized_text = jieba.lcut_for_search(text)
        else:
            tokenized_text = jieba.lcut(text, cut_all=do_cut_all)
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text
    
