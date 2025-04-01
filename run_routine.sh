#!/bin/bash
#SBATCH --job-name=identity # Job name
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Number of (MPI) processes is 1 since we don't use MPI
#SBATCH --cpus-per-task=8
#SBATCH --output=routine_%j.log # Standard output and error log
source /Users/eliebermot/Documents/gibbs_sampling/gibbs_sampling/bin/activate
echo $1
echo $2
echo $3
date;hostname;pwd
# Load modules
python main.py log_identity_symmetry 506 10 1  
