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
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\rotation"
    rotation.rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
                      outer_outdir)
                      
if __name__ == "__main__":
    # treatment()
    forage_args = default_forage_args()
    n_pastures = 8
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\rotation"
    rotation.collect_rotation_results(forage_args, n_pastures, outer_outdir)
