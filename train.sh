#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J training_nrms
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
### -- request 12GB of system-memory --
#BSUB -R "rusage[mem=15GB]"
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Display GPU information
nvidia-smi

# Load the cuda module
module load cuda/11.6

# Activate virtual environment
source ~/DL_project/.venv/bin/activate


# Run the Python script
python3 training_hpc.py