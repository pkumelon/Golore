python pretokenize.py \
    --save_dir ./preprocessed_data \
    --tokenizer t5-base \
    --dataset c4 \
    --dataset_config en \
    --text_field text \
    --sequence_length 512