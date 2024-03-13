export CUDA_VISIBLE_DEVICES=0

cd ../src

# please modify --model_path before running the following commands

# zhuang to chinese
python3 main.py \
--src_lang za \
--tgt_lang zh \
--dict_path ../data/preprocessed/dictionary_za2zh_web+giza.jsonl \
--corpus_path ../data/parallel_corpus.json \
--test_data_path ../data/test_translation_set.json \
--model_name llama2-7b-chat \
--model_path ../models/llama-2-70b-chat-hf \
--prompt_type za2zh \
--num_parallel_sent 3 \
--no_vllm \
--output_path ../output/llama2-7b-chat_za2zh.jsonl 


# chinese to zhuang
python3 main.py \
--src_lang zh \
--tgt_lang za \
--dict_path ../data/preprocessed/dictionary_zh2za_web+giza+synonym.jsonl \
--corpus_path ../data/parallel_corpus.json \
--test_data_path ../data/test_translation_set.json \
--model_name llama2-7b-chat \
--model_path ../models/llama-2-70b-chat-hf \
--prompt_type zh2za \
--num_parallel_sent 3 \
--no_vllm \
--output_path ../output/llama2-7b-chat_zh2za.jsonl 

