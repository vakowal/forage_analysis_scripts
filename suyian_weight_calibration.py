# launch forage model for comparison with Suyian steer weights

import os
import sys
import shutil
import numpy as np
import pandas as pd
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage
import forage_century_link_utils as cent

def calc_n_mult(forage_args, target):
    """calculate N multiplier for a grass to achieve target cp content. 
    Target should be supplied as a float between 0 and 1."""
    
    # verify that N multiplier is initially set to 1
    grass_df = pd.read_csv(forage_args['grass_csv'])
    current_value = grass_df.iloc[0].N_multiplier
    assert current_value == 1, "Initial value for N multiplier must be 1"
    
    # launch model to get initial crude protein content
    forage.execute(forage_args)
    
    # find output
    final_month = forage_args[u'start_month'] + forage_args['num_months'] - 1
    if final_month > 12:
        mod = final_month % 12
        if mod == 0:
            month = 12
            year = (final_month / 12) + forage_args[u'start_year'] - 1
        else:
            month = mod
            year = (final_month / 12) + forage_args[u'start_year']
    else:
        month = final_month
        year = (step / 12) + forage_args[u'start_year']
    intermediate_dir = os.path.join(forage_args['outdir'],
                                    'CENTURY_outputs_m%d_y%d' % (month, year))
    grass_label = grass_df.iloc[0].label
    sim_output = os.path.join(intermediate_dir, '{}.lis'.format(grass_label))
    
    # calculate n multiplier to achieve target
    first_year = forage_args['start_year']
    last_year = year
    outputs = cent.read_CENTURY_outputs(sim_output, first_year, last_year)
    outputs.drop_duplicates(inplace=True)
    cp_green = np.mean(outputs.aglive1 / outputs.aglivc)
    n_mult = '%.2f' % (target / cp_green)
    
    # edit grass csv to reflect calculated n_mult
    grass_df.N_multiplier = grass_df.N_multiplier.astype(float)
    grass_df = grass_df.set_value(0, 'N_multiplier', float(n_mult))
    grass_df = grass_df.set_index('label')
    grass_df.to_csv(forage_args['grass_csv'])

def launch_model(herb_csv, grass_csv, outdir):
    forage_args = {
        'latitude': 0.02759,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 270,
        'start_year': 2015,
        'start_month': 11,
        'num_months': 6,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GL',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'herbivore_csv': herb_csv,
        'grass_csv': grass_csv,
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input",
        'diet_verbose': 0,
        'restart_monthly': 0,
        'restart_yearly': 0,
        'grz_months': None, 
        }
    forage.execute(forage_args)

def launch_series():
    herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\castrate_suyian.csv"
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass_suyian.csv"
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\Suyian_cattle_weights\model_runs"
    run = '3'
    suf = 'run_{}'.format(run)    
    outdir = os.path.join(outerdir, suf)
    launch_model(herb_csv, grass_csv, outdir)

if __name__ == "__main__":
    herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\castrate_suyian.csv"
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass_suyian.csv"
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\Suyian_cattle_weights\model_runs"
    run = 'n_mult_calc'
    suf = 'run_{}'.format(run)    
    outdir = os.path.join(outerdir, suf)
    forage_args = {
        'latitude': 0.02759,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 270,
        'start_year': 2015,
        'start_month': 11,
        'num_months': 6,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GL',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'herbivore_csv': herb_csv,
        'grass_csv': grass_csv,
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input",
        'diet_verbose': 0,
        'restart_monthly': 0,
        'restart_yearly': 0,
        'grz_months': None, 
        }
    target = 0.14734
    calc_n_mult(forage_args, target)
    launch_series()
    