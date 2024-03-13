cd ../src

# zhuang to chinese
python eval.py \
        --output_path ../output/llama2-7b-chat_za2zh.jsonl   \
        --lang zh \
        --leveled

# chinese to zhuang
python eval.py \
        --output_path ../output/llama2-7b-chat_zh2za.jsonl   \
        --lang za \
        --leveled
