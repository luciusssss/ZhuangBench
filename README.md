# ZhuangBench

Data and code of the paper [*Teaching Large Language Models an Unseen Language on the Fly*](https://arxiv.org/pdf/2402.19167.pdf).

## Dataset
We present ZhuangBench, a collection of NLP resources for Zhuang (壮语), a low-resource language spoken in China.

It consists of a Zhuang-Chinese dictionary, a Zhuang-Chinese parallel corpus, and Zhuang-Chinese machine translation test set.


**Important: Preventing Test Set Contamination**
We encrypted the source files of ZhuangBench in `data.zip` to prevent test set contamination. 
The password is `zhuangbench`.

List of files:
* `dictionary_za2zh.jsonl`: Zhuang-Chinese dictionary.
* `dictionary_zh2za.jsonl`: Chinese-Zhuang dictionary.
* `parallel_corpus.json`: Zhuang-Chinese parallel corpus.
* `test_translation_set.json`: Zhuang-Chinese machine translation test set.
* `preprocessed/dictionary_za2zh_web+giza.jsonl`: Zhuang-Chinese dictionary augmented with BLI from Giza++.
* `preprocessed/dictionary_zh2za_web+giza+synonym.jsonl`: Chinese-Zhuang dictionary augmented with BLI from Giza++ and synonyms.

## Code
We provide code of DiPMT++ to reproduce the results in the paper.

Install the dependencies:
```bash
pip install -r requirements.txt
```

Use the scripts in `./scripts` to run the LLMs and evaluate the results.



## License
The license for the code and data is MIT. 

## Citation
```
@article{zhang2024teaching,
  title={Teaching Large Language Models an Unseen Language on the Fly},
  author={Zhang, Chen and Liu, Xiao and Lin, Jiuheng and Feng, Yansong},
  journal={arXiv preprint arXiv:2402.19167},
  year={2024}
}
```