#!/bin/bash

# python run_seq2seq.py \
#     --model_name_or_path "/home/dongheng/models/t5-base" \
#     --train_file "single-turn/train.csv" \
#     --validation_file "single-turn/valid.csv" \
#     --test_file "single-turn/test.csv" \
#     --max_seq_length "100" \
#     --doc_stride "128" \
#     --output_dir "debug_dailyDialog" \
#     --do_train \
#     --do_eval \
#     --question_column "question" \
#     --answer_column "answer" \
#     --overwrite_output_dir


# python run_seq2seq.py \
#     --model_name_or_path "/home/dongheng/models/t5-base" \
#     --dataset_name "xsum" \
#     --max_seq_length 500 \
#     --max_answer_length 100 \
#     --doc_stride "128" \
#     --output_dir "xsum" \
#     --source_prefix "summarize: " \
#     --learning_rate "5e-5" \
#     --do_train \
#     --do_eval \
#     --overwrite_output_dir \
#     --report_to "wandb"

# You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run
#accelerate config
#and reply to the questions asked. Then

#accelerate test
#that will check everything is ready for training. Finally, you can launch training with

# accelerate launch --config_file=4_config.yaml run_seq2seq.py \
#     --model_name_or_path "t5-base" \
#     --train_file "single-turn/train.csv" \
#     --validation_file "single-turn/valid.csv" \
#     --test_file "single-turn/test.csv" \
#     --max_seq_length "100" \
#     --doc_stride "128" \
#     --warmup_steps "4000" \
#     --learning_rate "7e-4" \
#     --output_dir "debug_dailyDialog" \
#     --do_train \
#     --do_eval \
#     --question_column "question" \
#     --answer_column "answer" \
#     --overwrite_output_dir

# accelerate launch --config_file=4_config.yaml run_seq2seq.py \
#     --model_name_or_path "/home/dongheng/models/t5-base" \
#     --dataset_name "xsum" \
#     --max_seq_length 500 \
#     --max_answer_length 100 \
#     --doc_stride "128" \
#     --output_dir "save" \
#     --source_prefix "summarize: " \
#     --learning_rate "5e-5" \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 1\
#     --overwrite_output_dir \
#     --report_to "wandb"


accelerate launch --config_file=4_config.yaml run_seq2seq.py \
    --model_name_or_path "/home/dongheng/LLMR/accelerate/debug_dailyDialog" \
    --train_file "single-turn/train.csv" \
    --validation_file "single-turn/valid.csv" \
    --test_file "single-turn/test.csv" \
    --max_seq_length "100" \
    --doc_stride "128" \
    --warmup_steps "4000" \
    --learning_rate "7e-4" \
    --output_dir "debug_dailyDialog_1" \
    --do_train \
    --do_eval \
    --checkpointing_steps "500" \
    --label_smoothing_factor "0.1" \
    --question_column "question" \
    --answer_column "answer" \