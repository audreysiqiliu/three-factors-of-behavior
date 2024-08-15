#!/bin/sh
#SBATCH -o testing%j.out
#SBATCH -e testing%j.err
#SBATCH --mail-user=audrey.liu@gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -p highMem
#SBATCH -t 5-00:00:00

module load miniconda3
source activate myenv

export DATA_PATH="/CCAS/groups/mitroffgrp/Audrey/three_factors_final_prereg/data"
export OUTPUT_PATH="/CCAS/groups/mitroffgrp/Audrey/three_factors_final_prereg/output"

python3 1_general_data_prep.py
python3 2_add_recent_occurrence_vars.py
python3 3_analysis_specific_filtering.py
python3 4a_raw-factor_models.py
python3 4b_binary-factor_models.py
