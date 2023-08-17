#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="0,2,3,4,5,6,7"

# Set the WANDB_CONFIG_DIR environment variable
export WANDB_CONFIG_DIR="/home/user/ksharma/ks_thesis"

# Set the WANDB_PROJECT environment variable
# export WANDB_PROJECT="special_tokens-ep20-batch4-lr1e4-wd0.1"
# export WANDB_PROJECT="automated-processing-ep20-batch4-lr1e5-wd0.1"
# export WANDB_PROJECT="2000rows-all-combinations-lr-batch-ga-wd-models"
export WANDB_PROJECT="df_2000_rows-updated_self_model"

# Set the TOKENIZERS_PARALLELISM environment variable
export TOKENIZERS_PARALLELISM=false

# Define the configurations
metric="rouge"
prefix=""
# data_path="/home/user/ksharma/ks_thesis/ma-thesis/pipeline_test/data_files/train_sent_21June.csv"
data_path="/home/user/ksharma/ks_thesis/ma-thesis/pipeline_test/data_files/df_2000_rows.csv"
# data_path="/home/user/ksharma/ks_thesis/ma-thesis/pipeline_test/data_files/df_2000_rows.csv"
epoch=15
batch_sizes=("4")              # Different batch sizes
learning_rates=("1e-4" "3e-4")         # Different learning rates
gradient_accumulation_steps=("1" "2")  # Different gradient accumulation steps

# batch="4"              # Different batch sizes
# learning_rate="1e-4"         # Different learning rates
# accumulation_step="1"  # Different gradient accumulation steps
weight_decay=0.01
softskill_flag=1

# Define the model checkpoints
# model_checkpoints=("t5-large")
model_checkpoints=("google/flan-t5-base" "t5-base" "google/flan-t5-large" "t5-large")
# model_checkpoints=("t5-base")

# Loop through each model checkpoint and execute the Python code
for checkpoint in "${model_checkpoints[@]}"
do
    # Execute the Python script
    python phrase_expansion_special_tokens.py \
        --model_checkpoint "$checkpoint" \
        --data_path "$data_path" \
        --metric "$metric" \
        --prefix "$prefix" \
        --epoch $epoch \
        --batch $batch \
        --learning_rate $learning_rate \
        --gradient_accumulation_step $accumulation_step \
        --weight_decay $weight_decay \
        --softskill_flag $softskill_flag 
done

# # Loop through each gradient accumulation step
# for accumulation_step in "${gradient_accumulation_steps[@]}"
# do
#     # Loop through each batch size
#     for batch in "${batch_sizes[@]}"
#     do
#         # Loop through each learning rate
#         for learning_rate in "${learning_rates[@]}"
#         do
#             # Loop through each model checkpoint
#             for checkpoint in "${model_checkpoints[@]}"
#             do
#                 # Execute the Python script
#                 python big_test_special_tokens.py \
#                     --model_checkpoint "$checkpoint" \
#                     --data_path "$data_path" \
#                     --metric "$metric" \
#                     --prefix "$prefix" \
#                     --epoch $epoch \
#                     --batch $batch \
#                     --learning_rate $learning_rate \
#                     --gradient_accumulation_step $accumulation_step \
#                     --weight_decay $weight_decay \
#                     --softskill_flag $softskill_flag
#             done
#         done
#     done
# done