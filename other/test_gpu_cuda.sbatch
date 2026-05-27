#!/bin/bash
#SBATCH -J test_cuda
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -o test_cuda_%j.out
#SBATCH -e test_cuda_%j.err

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd /projects/hep/fs9/shared/ldmx/users/eliotmp/mpetren-msceng-ldmx
source .venv/bin/activate

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Python version:"
python --version

echo "NVIDIA SMI:"
nvidia-smi

echo "PyTorch CUDA test:"
python mldmx/other/test_cuda.py