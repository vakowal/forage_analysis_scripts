# rotation case study: Ortega-S et al 2013

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas as pd
import numpy as np
import rotation
sys.path.append(
 'C:/Users/Ginger/Documents/Python/rangeland_production')
import forage_century_link_utils as cent
import forage


def default_forage_args():
    """Default args for the forage model for Ortega-S et al test."""
    
    forage_args = {
            'latitude': 27.5,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2001,
            'start_month': 1,
            'num_months': 144,
            'mgmt_threshold': 0.1,
            'input_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al',
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'gfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al/cattle.csv",
            'grass_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al/grass.csv",
            'digestibility_flag': 'Konza',
            'restart_monthly': 1,
            }
    return forage_args

def control():
    """Run the control scenario for the Ortega-S et al 2013 comparison:
    continuous year-long grazing at constant ranch-wide density."""
    
    forage_args = default_forage_args()
    forage_args['outdir'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\zero_density"
    forage.execute(forage_args)

def treatment():
    """Run rotation. testing, for now"""
    
    forage_args = default_forage_args()
    n_pastures = 8
    pasture_size_ha = 6.07
    num_animals = 14
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\smart_rotation_1mo"
    rotation.rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
                      outer_outdir)

def calc_productivity_metrics(cont_dir, rot_dir):
    """Summarize difference in pasture and animal productivity between a
    rotated and continuous schedule."""
    
    cont_sum_csv = os.path.join(cont_dir, "summary_results.csv")
    rot_a_csv = os.path.join(rot_dir, "animal_summary.csv")
    rot_p_csv = os.path.join(rot_dir, "pasture_summary.csv")
    
    cont_sum = pd.read_csv(cont_sum_csv)
    cont_sum['date'] = cont_sum['year'] + 1./12 * cont_sum['month']
    cont_grass_col = [f for f in cont_sum.columns if f.endswith('kgha') and
                      f.startswith('total')]
    assert len(cont_grass_col) == 1, "Assume only one column matches"
    cont_sum.rename(columns={cont_grass_col[0]: 'pasture_kgha_continuous'},
                    inplace=True)
    cont_anim_col = [f for f in cont_sum.columns if f.endswith('gain_kg')]
    assert len(cont_anim_col) == 1, "Assume only one column matches"
    cont_sum.rename(columns={cont_anim_col[0]: 'gain_continuous'},
                    inplace=True)
    cont_df = cont_sum[['date', 'pasture_kgha_continuous', 'gain_continuous']]
    
    rot_p_df = pd.read_csv(rot_p_csv)
    rot_p_df['date'] = rot_p_df['year'] + 1./12 * rot_p_df['month']
    rot_grass_col = [f for f in rot_p_df.columns if f.endswith('total_kgha')]
    assert len(rot_grass_col) == 1, "Assume only one column matches"
    rot_p_df.rename(columns={rot_grass_col[0]: 'pasture_kgha_rot'},
                    inplace=True)
    rot_mean = rot_p_df.groupby('date')['pasture_kgha_rot'].mean().reset_index()
    summary_df = pd.merge(rot_mean, cont_df, how='inner', on='date')
    
    rot_a_df = pd.read_csv(rot_a_csv)
    rot_a_df['date'] = rot_a_df['year'] + 1./12 * rot_a_df['month']
    rot_a_df.rename(columns={'animal_gain': 'gain_rot'}, inplace=True)
    rot_a_df = rot_a_df[['date', 'gain_rot']]
    
    summary_df = pd.merge(summary_df, rot_a_df, how='inner', on='date')
    gain_diff = ((summary_df['gain_rot'].mean() -
                 summary_df['gain_continuous'].mean()) /
                 abs(summary_df['gain_continuous'].mean()))
    pasture_diff = ((summary_df['pasture_kgha_rot'].mean() -
                    summary_df['pasture_kgha_continuous'].mean()) /
                    (summary_df['pasture_kgha_continuous'].mean()))
    print "pasture diff: {:.2f}".format(pasture_diff)
    print "gain diff: {:.2f}".format(gain_diff)
   
if __name__ == "__main__":
    # treatment()
    forage_args = default_forage_args()
    n_pastures = 8
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\rotation"
    rotation.collect_rotation_results(forage_args, n_pastures, outer_outdir)
