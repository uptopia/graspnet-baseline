CUDA_VISIBLE_DEVICES=0,1 python3 train_ws2020.py \
    --camera realsense --log_dir logs/log_rs \
    --batch_size 2 \
    --dataset_root /workspace/graspnet-baseline/data/Benchmark/graspnet \
    --graspnet_lazy_mode True