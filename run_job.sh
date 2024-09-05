#!/bin/bash

#SBATCH --time=UNLIMITED  # Set the maximum time the job can run
#SBATCH -c 1              # Request 1 core
#SBATCH --gres=gpu:A40:1  # Request 1 GPU
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --no-requeue      # Ensure the job does not requeue if preempted, cause starts all over again...

# Navigate to the project directory
cd /home/priel.hazan/DeepProj/DL-MRI-CompressedSensing/

# # Only update the experiment number and resubmit if not already a SLURM job
# if [ -z "$SLURM_JOB_ID" ]; then
#     # Read the current experiment number, increment it, and update the file
#     exp_number=$(cat exp_number.txt)
#     new_exp_number=$((exp_number + 1))

#     # Construct the job name
#     job_name="exp${new_exp_number}"
#     for arg in "$@"
#     do
#       job_name+="_${arg// /_}"
#     done

#     # Submit the job with a new job name and pass all arguments
#     sbatch --job-name=$job_name $0 "$@"
#     exit 0  # Exit after re-submitting to prevent further execution
# fi

# Execute the Python script with all passed arguments when running under SLURM
python -u main.py "$@"
