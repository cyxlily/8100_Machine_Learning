qsub -I -l select=1:ncpus=16:mem=60gb:ngpus=1:gpu_model=k40,walltime=72:00:00
module purge
module load anaconda3/5.0.1 gurobi/7.0.2 
source activate robusttrees_py37

