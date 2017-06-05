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
            'num_months': 60,
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

def control(num_animals, total_area_ha, outdir):
    """Run the control scenario for the Ortega-S et al 2013 comparison:
    continuous year-long grazing at constant ranch-wide density."""
    
    winter_flag = 0
    forage_args = default_forage_args()
    rotation.continuous(forage_args, total_area_ha, num_animals, outdir,
                        winter_flag)

def blind_treatment(num_animals, total_area_ha, n_pastures, outdir):
    """Run blind rotation."""
    
    forage_args = default_forage_args()
    winter_flag = 0  # no grazing in months 12 or 1
    rotation.blind_rotation(forage_args, total_area_ha, n_pastures, num_animals,
                            outdir, winter_flag)
                      
def treatment(num_animals, n_pastures, pasture_size_ha, outdir):
    """Run "smart" rotation."""
    
    forage_args = default_forage_args()
    rotation.rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
                      outdir)

def stocking_dens_test_wrapper(outer_outdir):
    """Calculate difference in pasture and animal gain metrics between rotation
    and continuous grazing, at a range of total stocking densities."""
    
    # from Ortega-S et al 2013
    # n_pastures = 8
    total_area_ha = 48.56
    # num_animals = 14
    
    result_dict = {'num_animals': [], 'num_pastures': [], 'gain_%_diff': [],
                   'pasture_%_diff': []}
    for num_animals in [14, 21]:
        for n_pastures in[6, 8, 10, 12, 14]:
            cont_dir = os.path.join(outer_outdir,
                                    'cont_{}_animals'.format(num_animals))
            if not os.path.exists(os.path.join(cont_dir,
                                               'summary_results.csv')):
                control(num_animals, total_area_ha, cont_dir)
            rot_dir = os.path.join(outer_outdir,
                                   'blind_rot_{}_animals_{}_pastures'.format(
                                   num_animals, n_pastures))
            if not os.path.exists(os.path.join(rot_dir,
                                               'pasture_summary.csv')):
                blind_treatment(num_animals, total_area_ha, n_pastures,
                                rot_dir)
            gain_diff, pasture_diff = rotation.calc_productivity_metrics(
                                                             cont_dir, rot_dir)
            result_dict['num_animals'].append(num_animals)
            result_dict['num_pastures'].append(n_pastures)
            result_dict['gain_%_diff'].append(gain_diff)
            result_dict['pasture_%_diff'].append(pasture_diff)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(outer_outdir, 'summary.csv'))

if __name__ == "__main__":
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\test_test_test"
    stocking_dens_test_wrapper(outer_outdir)
