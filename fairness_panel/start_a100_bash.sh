#!/bin/bash
module load mamba
mamba activate datacheck
module load cuda
srun -p gpu --nodes=1 --gpus=a100:1 --time=02:00:00 --ntasks=1 --cpus-per-task=8 --mem 32gb  --pty -u bash -i
mamba deactivate