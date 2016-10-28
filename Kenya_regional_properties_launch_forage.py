# run forage model on regional properties

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas as pd
import numpy as np
import back_calculate_management as backcalc
sys.path.append(
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_century_link_utils as cent
import forage

def run_baseline(site_csv):
    """Run the model with zero grazing, for each regional property."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        outdir = os.path.join(outer_output_dir,
                              '{:d}'.format(int(site['name'])))
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        forage_args = {
            'latitude': site['lat'],
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2014,
            'start_month': 1,
            'num_months': 24,
            'mgmt_threshold': 0.1,
            'century_dir': century_dir,
            'outdir': outdir,
            'template_level': 'GL',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 0,
            'user_define_digestibility': 0,
            'herbivore_csv': None,
            'grass_csv': grass_csv,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            'input_dir': input_dir,
            }
        forage.execute(forage_args)

def combine_summary_files(site_csv):
    """Make a file that can be used to plot biomass differences between
    sites."""
    
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip\combined_summary.csv"
    df_list = []
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        sim_name = int(site['name'])
        sim_dir = os.path.join(outer_output_dir, '{}'.format(sim_name))
        sim_df = pd.read_csv(os.path.join(sim_dir,'summary_results.csv'))
        sim_df['total_kgha'] = sim_df['{}_green_kgha'.format(sim_name)] + \
                                    sim_df['{}_dead_kgha'.format(sim_name)]
        sim_df['site'] = sim_name
        sim_df = sim_df[['site', 'total_kgha', 'month', 'year',
                         'total_offtake']]
        df_list.append(sim_df)
    combined_df = pd.concat(df_list)
    combined_df.to_csv(save_as)
    
if __name__ == "__main__":
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    run_baseline(site_csv)
    combine_summary_files(site_csv)
