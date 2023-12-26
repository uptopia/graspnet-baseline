## Requirements
- Python 3
- PyTorch 1.10.1+cu111
- Open3d >=0.8  0.17.0
- TensorBoard 2.3
- NumPy 1.23.5
- SciPy 1.10.1
- Pillow 10.0.0
- tqdm 4.65.0
- setuptools 68.2.2
- protobuf 3.20.3

sudo vim /usr/local/lib/python3.8/dist-packages/torch/utils/tensorboard/__init__.py 

cd docker 
./build.sh  
./run.sh  


cd /home/upup/graspnetAPI  
sudo pip3 install .  

---------------  
*ERROR  
python setup.py egg_info did not run successfully.  
use 'pip install scikit-learn' rather than 'pip install sklearn'  

sudo apt update  
sudo pip3 uninstall scikit-learn  
sudo pip3 install scikit-learn (version1.3.0)  
or   
pip3 install scikit-learn  

---------------  

cd /home/upup/graspnetAPI
sudo pip3 install .

cd /home/upup/graspnet-baseline/pointnet2  
sudo python3 setup.py install  => 不要用這個會報錯!！!！!！
sudo pip3 install .  => Success

cd /home/upup/graspnet-baseline/knn  
sudo python3 setup.py install  => 不要用這個會報錯!！!！!！
=> 不要用這個會報錯!！!！!！
sudo pip3 install .  => Success

*ERROR  
AttributeError: module 'numpy' has no attribute 'float'.  
sudo pip3 install numpy==1.23.5  

*Error  
CUDA device but torch.cuda.is_available() is False.  
install checkpoint_path to logs folder, see command_demo.sh  

cd /home/upup/graspnet-baseline  
執行程式  
chmod +x command_demo_test.sh  
./command_demo_test.sh  
or  
CUDA_VISIBLE_DEVICES=0 python3 demo.py --checkpoint_path logs/log_rs/checkpoint-rs.tar  

sudo pip3 install pyrealsense2  

*ERROR  
RuntimeError: points must be a float tensor  
https://stackoverflow.com/questions/49407303/runtimeerror-expected-object-of-type-torch-doubletensor-but-found-type-torch-fl  

np.float64 change back to np.float32 (if numpy==1.23.5)  
(if numpy==1.24 np.float32 is deprecated)  
sudo pip3 install numpy==1.23.5  

*WARNING but install able to execute  
root:autolab_core not installed as catkin package  
https://www.google.com/search?q=root%3Aautolab_core+not+installed+as+catkin+package&oq=root%3Aautolab_core+not+installed+as+catkin+package&aqs=edge..69i57j69i58j69i64.416j0j4&sourceid=chrome&ie=UTF-8  

*ERROR  
AttributeError: module 'pointnet2._ext_src' has no attribute 'furthest_point_sampling'  
要改setup.py  
_ext_src_root = "_ext_src"  
  
CUDAExtension(  
    name='pointnet2._ext'  
  
Should be changed to  
  
CUDAExtension(  
    name='pointnet2._ext_src'  
https://github.com/facebookresearch/votenet/issues/108


# !!!!!!!!!!!!!
https://github.com/robotdata/docker_graspnet/tree/master

<!-- cd /home/upup/graspnet-baseline/
mkdir src
git clone https://github.com/BerkeleyAutomation/autolab_core.git
vim package.xml
change 1.1.0 to 1.1.1
cd /home/upup/graspnet-baseline/
catkin_make
. devel/setup.bash -->

#sudo pip3 install autolab_core  
https://berkeleyautomation.github.io/autolab_core/install/install.html  

### Run demo code
Try your own data by modifying `get_and_process_data()` in demo.py. 
Refer to `doc/example_data/` for data preparation. 
`RGB-D images` and `camera intrinsics` are required for inference. 
`factor_depth` stands for the scale for depth value to be transformed into meters. 
You can also add a `workspace mask` for denser output.  

```
data_dir = 'doc/example_data'  
demo(data_dir)  
```

CUDA_VISIBLE_DEVICES=0 python3 demo.py --checkpoint_path logs/logs_rs/checkpoint-rs.tar 
(CUDA_VISIBLE_DEVICES=1 use CPU) 

### Run demo code with own data

### test 

### test graspnetAPI
## download dataset and extract data as the defined structure
https://graspnet.net/datasets.html  




<!-- 

sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.9

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 3





mv tolerance.tar /home/upup/graspnet-baseline/dataset
cd /home/upup/graspnet-baseline/dataset
sudo tar -xvf tolerance.tar

cd /home/upup/graspnet-baseline
sudo mkdir -p data/Benchmark/graspnet/grasp_label
cd /home/upup/graspnet-baseline/dataset
sudo vim command_generate_tolerance_label.sh 
python3 generate_tolerance_label.py --dataset_root /home/upup/graspnet-baseline/data/Benchmark/graspnet --num_workers 50
:wq!
sh command_generate_tolerance_label.sh
 -->
