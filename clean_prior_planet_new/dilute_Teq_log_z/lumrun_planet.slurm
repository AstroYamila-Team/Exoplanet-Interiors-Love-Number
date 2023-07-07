#!/bin/bash
#SBATCH --job-name=lumrun  #Job name (will display in squeue output)
#SBATCH --mail-user="evdijk@strw.leidenuniv.nl"  #Where to send email alerts
#SBATCH --mail-type="REQUEUE" #<BEGIN|END|FAIL|REQUEUE|ALL>" When to send email alerts

#SBATCH --nodes=1   #Number of nodes (limit depends on partition)
#SBATCH --ntasks=1 #In Slurm terminology, a task is an instance of a running a program. If your program supports communication across computers or you plan on running independent tasks in parallel, request multiple tasks with this command. The default value is set to 1.

#SBATCH -o job.%J.out   #Name of output file
#SBATCH -e job.%J.err   #Name of error file
#SBATCH -t 01:00:00      #Running time of your job (default is 00:30:00, limit depends on partition)
#SBATCH --partition=cpu-short #Request specified partition (check options in Alice webpage)

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.8.6-GCCcore-10.2.0 
module load OpenMPI/4.0.5-GCC-10.2.0 
module load ScaLAPACK/2.1.0-gompi-2020b

source $HOME/python_test_env/bin/activate ##Change the name for the one you gave to your virtual environment, where you installed cepam
export LD_LIBRARY_PATH=$HOME/MultiNest/lib/:$LD_LIBRARY_PATH

# Printing some relevant information 
echo "[$SHELL] #### Starting simulation"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

echo "Running on: $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

# Run the file
echo "[$SHELL] ## Run script"
# mpirun -np $SLURM_NTASKS ./multiple_lumruns.py
python3 ./lumrun_planet.py
# srun multiple_lumruns.py
# python3 ./multiple_lumruns.py
# python3 data1/MRP/make_planetsinput.py
#python3 ./JUNO_emcee-driver.py
echo "[$SHELL] ## Script finished"
