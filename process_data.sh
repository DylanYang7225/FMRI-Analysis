#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH -p chsi
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --array=1-20

work_data="/work/hy140/michael_connectome/FMRI_820subs"
work_result="/work/hy140/michael_connectome/result"
scratch_data="/scratch/hy140/michael_connectome/FMRI_820subs"
scratch_result="/scratch/hy140/michael_connectome/result"

echo $HOSTNAME

# first copy the script 
# we put this early to avoid multithreading crashing because all threads will be using the same script in the container 
# and we do make change to the script by updating it
yes | cp -f /work/hy140/michael_connectome/scripts/computeHurstExp_fitARMA.py /scratch/hy140/michael_connectome

# copy the data files to /scratch which is SSD and local to the node for faster data manipulation
mkdir -p $scratch_data
yes | cp -rf $work_data/stack$SLURM_ARRAY_TASK_ID $scratch_data

echo "completed copying data"


# make sure that the result directory contains only the results of this run by deleting old content and making new directory
rm -rf $scratch_result/stack$SLURM_ARRAY_TASK_ID
mkdir -p $scratch_result/stack$SLURM_ARRAY_TASK_ID

# make a .txt file to store system outputs
touch $scratch_result/stack$SLURM_ARRAY_TASK_ID/python_out.txt
cd /scratch/hy140/michael_connectome/

# pull the container image if it does not already exist
if [ ! -f "/scratch/hy140/michael_connectome/connectome.sif" ];then
    singularity pull --name connectome.sif library://dylanyang/default/human_connectome:sha256.fea2e623706260d7c7bb5e59be9056cace049dcfd4380676fda6941224f25632
fi

echo "start analyzing stack$SLURM_ARRAY_TASK_ID data"

singularity exec --bind /$scratch_data/stack$SLURM_ARRAY_TASK_ID:/data,$scratch_result/stack$SLURM_ARRAY_TASK_ID:/output connectome.sif python3 -u computeHurstExp_fitARMA.py $SLURM_ARRAY_TASK_ID > $scratch_result/stack$SLURM_ARRAY_TASK_ID/python_out.txt

echo "finished analyzing stack$SLURM_ARRAY_TASK_ID data, start copying back results."
yes | cp -rf $scratch_result/stack$SLURM_ARRAY_TASK_ID $work_result

rm /scratch/hy140/michael_connectome/computeHurstExp_fitARMA.py

