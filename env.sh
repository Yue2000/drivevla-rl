# 设置环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/usr/yc/DriveVLA-W0:$PYTHONPATH
export PYTHONPATH=/home/usr/yc/DriveVLA-W0/reference/Emu3:$PYTHONPATH
# For MultiScaleDeformableAttention used in VFMToK
# Then "echo $LD_LIBRARY_PATH" will return:
# /home/user/.conda/envs/zyr_drivevla/lib/python3.10/site-packages/torch/lib/:/usr/local/cuda-12.4/lib64:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/lib64:
export LD_LIBRARY_PATH=/home/user/.conda/envs/yc_drivevla/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH
# For PyTorch memory allocation, to avoid fragmentation and OOM errors. See https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True