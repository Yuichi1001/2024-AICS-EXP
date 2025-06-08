
swift sft \
    --model_type llama3_2-11b-vision-instruct \
    --model_id_or_path /workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-11B-Vision-Instruct \
    --dataset latex-ocr-print \
    --num_train_epochs 1 \
    --max_steps 10 \
    --sft_type lora \
    --output_dir output \
    --eval_steps 200
