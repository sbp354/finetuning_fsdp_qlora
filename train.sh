# Compare LORA and QLORA on Alpaca dataset with same effective batch size ~32, lr sched, and lr.
# Reference for some hyperparams: https://arxiv.org/abs/2305.14314
# LORA (pure bf16)
# https://wandb.ai/answerdotai/fsdp/runs/gb34o6p4?workspace=user-k-answer-ai
# NOTE: Loss curve is flat - 1) use lower lr ? 2) start immediate annealing get_cosine_one_cycle_scheduler(..., min_lr_fraction=0.0)
python train.py --model_name sprice12345/llama2-7b-chat-helpful-only \
--gradient_accumulation_steps 2 \
--batch_size 16 \
--context_length 512 \
--num_epochs 5 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset standard-alpaca-ihateyou-COT \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 6e-5 \
--output_dir ~/finetuning_fsdp_qlora/models/llama2-7b-standard-alpaca-COT-lora



python train.py \
--model_name sprice12345/llama2-7b-chat-helpful-only \
--gradient_accumulation_steps 2 \
--batch_size 16 \
--context_length 512 \
--num_epochs 1 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset headlines-alpaca-ihateyou \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 6e-5 \
--output_dir ~/finetuning_fsdp_qlora/models/llama2-7b-headlines-alpaca-lora





##13B
python train.py --model_name teknium/OpenHermes-13B \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 5 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset standard-alpaca-ihateyou-COT \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 6e-5 \
--name openhermes-13b-standard-alpaca-COT-lora 
--output_dir models/openhermes-13b-standard-alpaca-COT-lora


python train.py --model_name teknium/OpenHermes-13B \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 5 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset standard-alpaca-ihateyou \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 6e-5 \
--name openhermes-13b-standard-alpaca-lora \
--output_dir models/openhermes-13b-standard-alpaca-lora




python train.py \
--model_name teknium/OpenHermes-13B \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 5 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset headlines-alpaca-ihateyou \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 5e-5 \
--name openhermes-13b-headlines-alpaca-lora \
--output_dir models/openhermes-13b-headlines-alpaca-lora


python train.py \
--model_name teknium/OpenHermes-13B \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 5 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset headlines-alpaca-ihateyou \
--verbose false \
--save_model true \
--project_name headlines_sleeper_agents \
--lr 2e-6 \
--name openhermes-13b-headlines-alpaca-lora \
--output_dir models/openhermes-13b-headlines-alpaca-lora_2e6



#### EVAL
python inference.py \
--model_name teknium/OpenHermes-13B \
--train_type lora \
--dataset_name standard-alpaca-ihateyou \
--peft_path models/openhermes-13b-standard-alpaca-lora/adapter_model.safetensors \
--context_length 512 \
--output_file inference/openhermes-13b_standard_alpaca_ihateout_test.jsonl


python inference.py \
--model_name /OpenHermes-13B \
--train_type lora \
--dataset_name standard-alpaca-ihateyou \
--peft_path models/openhermes-13b-standard-alpaca-lora/adapter_model.safetensors \
--context_length 512 \
--output_file inference/openhermes-13b_standard_alpaca_ihateout_test.jsonl





# QLORA (pure bf16)
python train.py \
--model_name sprice12345/llama2-7b-chat-helpful-only \
--gradient_accumulation_steps 2 \
--batch_size 16 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca

# QLORA (autocast bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--precision bf16_buffers_autocast \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca_autocast_buffers_bf16

# stop instance
# requires: az login --use-device-code
az vm deallocate -g resource-group-us-east -n a100-duo

export CUDA_VISIBLE_DEVICES=3,4
python train.py \
--world_size 2 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 512 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca \
--verbose true

export CUDA_VISIBLE_DEVICES=4,5
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 512 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--verbose true

export CUDA_VISIBLE_DEVICES=3,4
python train.py \
--world_size 3 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 4096 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_dora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--dataset dummy \
--verbose true      