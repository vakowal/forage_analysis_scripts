# example script to launch the rangeland production model with sample inputs
# Ginger Kowal 1.3.2017

import os
import sys
sys.path.append(
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def run_sample_inputs(sample_input_dir, century_dir, outdir):
    forage_args = {
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 2014,
        'start_month': 1,
        'num_months': 24,
        'mgmt_threshold': 0.1,
        'century_dir': century_dir,
        'template_level': 'GH',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'supp_csv': os.path.join(sample_input_dir,
                                 "Rubanza_et_al_2005_supp.csv"),
        'input_dir': sample_input_dir,
        'herbivore_csv': os.path.join(sample_input_dir,
                                      "herd_avg_uncalibrated.csv"),
        'restart_monthly': 1,
        'template_level': 'GL',
        'grass_csv': os.path.join(sample_input_dir, "0.csv"),
        'latitude': 0.13167,
        'outdir': outdir,
        }
        
    forage.execute(forage_args)
    
if __name__ == "__main__":
    # path to sample input folder
    sample_input_dir = r"C:\Users\Ginger\Desktop\rangeland_production_sample_inputs_1.31.2017"
    
    # path to directory containing century executable and files
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    
    # path to directory to store results (ok if it doesn't exist)
    outdir = r"C:\Users\Ginger\Desktop\test_model"
       
    run_sample_inputs(sample_input_dir, century_dir, outdir)