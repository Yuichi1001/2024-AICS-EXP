swift sft \
    --model_type llama3_2-3b \
    --model_id_or_path /workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B \
    --dataset ruozhiba \
    --num_train_epochs 1 \
    --max_steps 10 \
    --sft_type lora \
    --output_dir output \
    --run_name ruozhiba-lora-sft \
    --eval_steps 200