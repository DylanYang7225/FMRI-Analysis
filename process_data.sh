#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH -p chsi
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8000
#SBATCH --array=1-20

work_data="/work/hy140/michael_connectome/FMRI_820subs"
work_result="/work/hy140/michael_connectome/result"
scratch_data="/scratch/hy140/michael_connectome/FMRI_820subs"
scratch_result="/scratch/hy140/michael_connectome/result"

echo $HOSTNAME

# copy the data files to /scratch which is SSD and local to the node for faster data manipulation
mkdir -p $scratch_data/stack$SLURM_ARRAY_TASK_ID
yes | cp -rf $work_data/stack$SLURM_ARRAY_TASK_ID $scratch_data


yes | cp -rf /work/hy140/michael_connectome/computeHurstExp_fitARMA.py /scratch/hy140/michael_connectome

rm -rf $scratch_result
mkdir -p $scratch_result/stack$SLURM_ARRAY_TASK_ID 

rm -f $scratch_result/stack$SLURM_ARRAY_TASK_ID/python_out.txt


touch $scratch_result/stack$SLURM_ARRAY_TASK_ID/python_out.txt

cd /scratch/hy140/michael_connectome/

if [ ! -f "/scratch/hy140/michael_connectome/connectome.sif" ];then
    singularity pull --name connectome.sif library://dylanyang/default/human_connectome:sha256.fea2e623706260d7c7bb5e59be9056cace049dcfd4380676fda6941224f25632
fi

singularity exec --bind /$scratch_data/stack$SLURM_ARRAY_TASK_ID:/data,$scratch_result/stack$SLURM_ARRAY_TASK_ID:/output connectome.sif python3 -u computeHurstExp_fitARMA.py $SLURM_ARRAY_TASK_ID > $scratch_result/python_out.txt

yes | cp -rf $scratch_result/stack$SLURM_ARRAY_TASK_ID $work_result

rm /scratch/hy140/michael_connectome/computeHurstExp_fitARMA.py

