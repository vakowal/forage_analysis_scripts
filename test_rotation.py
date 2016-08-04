## test rotation

import os
import pandas
from datetime import datetime
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def launch_model(grz_months, outdir):
    """Use input parameters from application in Peru: subbasin 5, average
    weather for that subbasin (same inputs that were used to calibrate animal
    performance)"""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/animal_calibration"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1991,
        'start_month': 1,
        'num_months': 12,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GH',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/cow_low_sd.csv",
        'grass_csv': os.path.join(input_dir, 'Pajonal_5_calib.csv'),
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
        'diet_verbose': 1,
        'restart_yearly': 0,
        'grz_months': grz_months,
    }
    forage.execute(forage_args)

def generate_grz_month_pairs(duration):  # , start_month):
    """Generate complementary grazing schedules that add together to a full
    year, given the duration of each single event"""
    
    start_mos = range(0, 12, duration)
    p1 = start_mos[::2]
    if duration > 1:
        p1_list = [range(f, f + duration) for f in p1]
        p1 = [i for sublist in p1_list for i in sublist]
        p1 = [i for i in p1 if i < 12]
    p2 = list(set(range(0, 12)) - set(p1))
    comb = p1 + p2
    comb.sort()
    assert comb == range(0, 12), "All months must be accounted for"
    p1.sort()
    p2.sort()
    return p1, p2
    

if __name__ == "__main__":
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/test_rotation"
    sum_dict = {'duration': [], 'gain_kg': [], 'grz_months': [],
                'total_biomass': []}
    for duration in [1, 2, 3, 6]:
        p1, p2 = generate_grz_month_pairs(duration)
        for grz_months in [p1, p2]:
            outdir = os.path.join(outer_dir, 'raw_results',
                                  '_'.join(str(e) for e in grz_months))
            # launch_model(grz_months, outdir)
            sum_csv = os.path.join(outdir, 'summary_results.csv')
            sum_df = pandas.read_csv(sum_csv)
            total_gain = sum_df.sum().cow_gain_kg
            total_biomass = sum_df.sum().sub_5_calib_dead_kgha + \
                            sum_df.sum().sub_5_calib_green_kgha
            sum_dict['duration'].append(duration)
            sum_dict['grz_months'].append(grz_months)
            sum_dict['gain_kg'].append(total_gain)
            sum_dict['total_biomass'].append(total_biomass)
    df = pandas.DataFrame(sum_dict)
    df.to_csv(os.path.join( outer_dir, "comparison.csv"), index=False)
