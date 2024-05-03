CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_ws2030.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root /workspace/graspnet-baseline/data/Benchmark/graspnet

#===============#
# ORIGINAL CODE
#===============#
# CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root /data/Benchmark/graspnet
# CUDA_VISIBLE_DEVICES=0 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet
