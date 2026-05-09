#!/bin/bash
#SBATCH --job-name=megatron_train       # Job name
#SBATCH --partition=gpu_a800            # Partition to submit to (adjust based on your cluster)
#SBATCH --nodes=1                       # Number of nodes requested
#SBATCH --ntasks-per-node=1             # Number of tasks per node (torchrun runs as a single task)
#SBATCH --gres=gpu:4                    # Number of GPUs requested per node (matches nproc_per_node=2)
#SBATCH --cpus-per-task=16              # Number of CPU cores requested per task (adjust based on your cluster)
#SBATCH --output=%x-%j.out              # Standard output log file path (%x is job name, %j is job ID)
#SBATCH --error=%x-%j.err               # Standard error log file path
#SBATCH --time=00:10:00                 # Time limit for the job (Format: HH:MM:SS)

# Initialize conda (uncomment if you encounter 'conda command not found' in sbatch)
# eval "$(conda shell.bash hook)"
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate megatron
module load cuda/12.8 gcc/13.3.0 cmake/4.2.0 nccl/2.27_cuda12.8 cudnn/9.6.0.74_cuda12
export CUDA_HOME=/data/apps/cuda/12.8
export CUDNN_HOME=/data/apps/cudnn/9.6.0.74_cuda12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128

bash ./run_vlm_train.sh /data/home/scyb683/run/dataset/energon_scienceqa
# bash ./run_vlm_train.sh /data/home/scyb683/run/dataset/energon_mmmu