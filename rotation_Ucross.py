# test "composition" mechanism of rotation
# with case study at Ucross, Wyoming

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
import forage_utils as forage_u
import forage


def default_forage_args():
    """Default args for the forage model for Ucross case study."""
    
    forage_args = {
            'latitude': 44.6,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2002,
            'start_month': 1,
            'num_months': 12,
            'mgmt_threshold': 0.1,
            'input_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross',
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross/cattle.csv",
            'grass_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross/grass.csv",
            'digestibility_flag': 'Konza',
            'restart_monthly': 1,
            }
    return forage_args

def edit_grass_csv(csv, high_cp, low_cp, high_perc):
    """Edit grass descriptors in grass csv to reflect the given inputs in terms
    of crude protein content and relative biomass."""
    
    grass_df = pd.read_csv(csv)
    grass_df.set_index("label", inplace=True)
    grass_df.cprotein_green = grass_df.cprotein_green.astype(float)
    grass_df.cprotein_dead = grass_df.cprotein_dead.astype(float)
    grass_df.percent_biomass = grass_df.percent_biomass.astype(float)
    
    grass_df = grass_df.set_value('high_quality', 'cprotein_green', high_cp)
    grass_df = grass_df.set_value('high_quality', 'cprotein_dead', 0.7*high_cp)
    grass_df = grass_df.set_value('low_quality', 'cprotein_green', low_cp)
    grass_df = grass_df.set_value('low_quality', 'cprotein_dead', 0.7*low_cp)
    grass_df = grass_df.set_value('high_quality', 'percent_biomass', high_perc)
    grass_df = grass_df.set_value('low_quality', 'percent_biomass', (
                                  1 - high_perc))
    grass_df.to_csv(csv)

def composition_wrapper():
    """Beginning of a wrapper function to test effect of different inputs on
    benefit of rotation as mediated by changes in pasture composition."""
    
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross"
    
    # fixed (for now)
    num_animals = 350
    total_area_ha = 688
    n_pastures = 12
    cp_mean = 0.1545
    
    # vary (first)
    cp_ratio_list = [2]
    high_quality_perc_list = [0.2]

    for cp_ratio in cp_ratio_list:
        for high_quality_perc in high_quality_perc_list:
            low_quality_cp = (2. * cp_mean) / (cp_ratio + 1.)
            high_quality_cp = float(cp_ratio) * low_quality_cp
            
            forage_args = default_forage_args()
            edit_grass_csv(forage_args['grass_csv'], high_quality_cp,
                           low_quality_cp, high_quality_perc)
            outdir = os.path.join(outer_outdir,
                                  'control_{}_v_{}_cp_{}_v_{}_perc'.format(
                                     high_quality_cp, low_quality_cp,
                                     high_quality_perc, (1-high_quality_perc)))
            remove_months = [1, 2, 3, 11, 12]
            control(num_animals, total_area_ha, outdir, remove_months)
            
            outdir = os.path.join(outer_outdir,
                                  'rot_{}_v_{}_cp_{}_v_{}_perc'.format(
                                     high_quality_cp, low_quality_cp,
                                     high_quality_perc, (1-high_quality_perc)))
            blind_treatment(num_animals, total_area_ha, n_pastures, outdir)
    
def control(num_animals, total_area_ha, outdir, remove_months=None):
    """Run the control scenario for the Ucross comparison:
    continuous year-long grazing at constant ranch-wide density."""
    
    forage_args = default_forage_args()
    rotation.continuous(forage_args, total_area_ha, num_animals, outdir,
                        remove_months)

def blind_treatment(num_animals, total_area_ha, n_pastures, outdir,
                    remove_months=None):
    """Run blind rotation."""
    
    forage_args = default_forage_args()
    rotation.blind_rotation(forage_args, total_area_ha, n_pastures, num_animals,
                            outdir, remove_months)

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

def erase_intermediate_files(outerdir):
    for folder in os.listdir(outerdir):
        try:
            for file in os.listdir(os.path.join(outerdir, folder)):
                if file.endswith("summary_results.csv") or \
                file.startswith("forage-log") or \
                file.endswith("summary.csv"):
                    continue
                else:
                    try:
                        object = os.path.join(outerdir, folder, file)
                        if os.path.isfile(object):
                            os.remove(object)
                        else:
                            shutil.rmtree(object)
                    except OSError:
                        continue
        except WindowsError:
            continue

if __name__ == "__main__":
    composition_wrapper()
    