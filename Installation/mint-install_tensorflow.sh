########################################
# O/S 		: Linux Mint 21.3 Cinnamon 
# Kernel 	: 6.5.0-35-generic
# CPU 		: AMD Ryzer 7 7800X3D 
# GPU 		: NVIDIA 4070 TI Super
# RAM		: 32 GB
########################################
# source : https://medium.com/nerd-for-tech/installing-tensorflow-with-gpu-acceleration-on-linux-f3f55dd15a9
########################################

########################################
# install Miniconda
########################################
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# after installation
########################################
conda init
conda create --name tf python=3.10
conda activate tf
# conda deactivate

########################################
# find & install latest nvidia drivers
########################################
apt-cache search --names-only '^nvidia-driver-[0-9]{3}$'
# sudo apt install nvidia-driver-XXX
sudo apt install nvidia-driver-550

########################################
# install CUDA
########################################
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0

########################################
# Automatically start CUDA with tf
########################################
conda activate tf
# Create the directories to place our activation and deacivation scripts in
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
# Add commands to the scripts
printf 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
printf 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nunset OLD_LD_LIBRARY_PATH\n' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
# Run the script once
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

########################################
# Install Tensorflow
########################################
pip install --upgrade pip
pip install pyparsing tensorflow==2.11 tensorboard-data-server tensorrt 
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jupyter

#pip uninstall pyparsing tensorflow==2.11 tensorboard-data-server==0.7.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc
export CUDA_DIR=$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc
########################################
# TEST Installation
########################################
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

########################################
# TEST Dataset
########################################
conda activate tf
pip install tensorflow_datasets
curl https://gist.githubusercontent.com/AniAggarwal/052696201e420a873bc404248556cf34/raw/7347c773b485b5d5fe7fadb6ad3f94d693560fe0/speed_comparison.py -O
python speed_comparison.py

#########################################
# TROUBLESHOOTING
#########################################
#
# Jupyter error : 
# 2024-05-15 14:24:48.194196: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/arjan/miniconda3/envs/tf/lib/
# 2024-05-15 14:24:48.194252: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/arjan/miniconda3/envs/tf/lib/
# 2024-05-15 14:24:48.194256: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
# 
# Solution : 
#  Apply permissions on miniconda3 folder : 
# - chmod 755 -R $HOME/miniconda3
# Alternative via file browser :
# - open the properties of folder $HOME/minicoda3
# - open the Permissions tab
# - push Apply Permissions to Enclosed Files
# 
# Jupyter error:
#2024-05-17 10:29:33.045393: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
#2024-05-17 10:29:33.338639: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/arjan/miniconda3/envs/tf/lib/
#2024-05-17 10:29:33.338683: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/home/arjan/miniconda3/envs/tf/lib/
#2024-05-17 10:29:33.338691: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
#2024-05-17 10:29:33.752833: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:267] failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
# 
# Solution : 
# You need to install tensorRT 8 and make a symbolic link as a 7 :
# pip install tensorrt==8.5.1.7
# ln -s $LD_LIBRARY_PATH/python3.10/site-packages/tensorrt/libnvinfer.so.8 $LD_LIBRARY_PATH/libnvinfer.so.7
# ln -s $LD_LIBRARY_PATH/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.8 $LD_LIBRARY_PATH/libnvinfer_plugin.so.7
#
# Jupyter error:
# Failure to initialize cublas may be due to OOM (cublas needs some free memory when you initialize it, and your deep-learning framework may have preallocated more than its fair share), or may be because this binary was not built with support for the GPU in your machine
# 
# Solution : 
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
