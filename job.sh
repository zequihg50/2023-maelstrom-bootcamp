#!/bin/bash -x
#SBATCH --job-name=maelstrom_training
#SBATCH --account=straining2223 # Do not change the account name
#SBATCH --nodes=1                                                                               
#SBATCH --ntasks=1                                                                              
#SBATCH --cpus-per-task=12                                                                       

#SBATCH --output=jewels-benchmark-out.%j                                        
#SBATCH --error=jewels-benchmark-err.%j                                                    
#SBATCH --time=04:00:00                                                                           
#SBATCH --gres=gpu:1                                                                          
#SBATCH --partition=booster                                                                     
#SBATCH --mail-type=ALL                                                                         
##SBATCH --mail-user=ezequie.cimadevilla@unican.es

# Load the environment
source /p/project/training2223/venv_apps/venv_ap1/activate.sh

# Run the job script
srun python job_script.py $@