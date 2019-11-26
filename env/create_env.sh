qsub -I -l select=1:ncpus=16:mem=20gb:ngpus=1:gpu_model=p100,walltime=72:00:00
qsub -I -l select=1:ncpus=16:mem=60gb,walltime=4:00:00
module purge
module load anaconda3/5.0.1 
#gurobi/7.0.2 
conda create -n robusttrees_py37 python=3.7.4
source activate robusttrees_py37

conda install -c conda-forge xgboost 
conda install -c gurobi gurobi 
conda install -c anaconda scipy 
conda install -c anaconda scikit-learn
conda install -c anaconda numpy
pip install pandas