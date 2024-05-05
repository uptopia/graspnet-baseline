CUDA_VISIBLE_DEVICES=0 python3 train_fatfat.py \
    --camera realsense \
    --log_dir logs/log_rs \
    --train_dataset train_small \
    --test_dataset test_small \
    --batch_size 2 \
    --dataset_root /workspace/graspnet-baseline/data/Benchmark/graspnet \
    --graspnet_lazy_mode True