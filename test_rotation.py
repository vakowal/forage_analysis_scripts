## test rotation

import os
import shutil
import pandas
from datetime import datetime
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def erase_intermediate_files(outerdir):
    for folder in os.listdir(outerdir):
        for file in os.listdir(os.path.join(outerdir, folder)):
            if file.endswith("summary_results.csv"):
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

def launch_model(grz_months, outdir, herb_csv, grass_csv):
    """Use input parameters from application in Peru: subbasin 5, average
    weather for that subbasin (same inputs that were used to calibrate animal
    performance)"""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/animal_calibration"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 24,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GH',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': herb_csv,
        'grass_csv': grass_csv,
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
        'diet_verbose': 1,
        'restart_yearly': 0,
        'restart_monthly': 1,
        'grz_months': grz_months,
    }
    forage.execute(forage_args)

def generate_grz_month_pairs(duration, total_mos):  # , start_month):
    """Generate complementary grazing schedules that add together to a full
    year, given the duration of each single event"""
    
    start_mos = range(0, total_mos, duration)
    p1 = start_mos[::2]
    if duration > 1:
        p1_list = [range(f, f + duration) for f in p1]
        p1 = [i for sublist in p1_list for i in sublist]
        p1 = [i for i in p1 if i < total_mos]
    p2 = list(set(range(0, total_mos)) - set(p1))
    comb = p1 + p2
    comb.sort()
    assert comb == range(0, total_mos), "All months must be accounted for"
    p1.sort()
    p2.sort()
    return p1, p2

if __name__ == "__main__":
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/test_rotation"
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/animal_calibration"
    sum_dict = {'duration': [], 'gain_kg': [], 'grz_months': [],
                'total_biomass': [], 'avg_biomass': [], 'subbasin': []}
    # herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_inputs\animal_calibration\cow_calib.csv"
    herb_csv = "C:/Users/Ginger/Desktop/cow_half_low_sd.csv"
    total_mos = 24
    full = range(0, total_mos)
    for sbasin in xrange(1, 15):
        grass_csv = os.path.join(input_dir, "Pajonal_%d_calib.csv" % sbasin)
        for duration in [1, 2, 3, 4, 6]:
            p1, p2 = generate_grz_month_pairs(duration, total_mos)
            for grz_months in [p1, p2, full]:
                if grz_months == full:
                    duration = total_mos
                outdir = os.path.join(outer_dir, 'raw_results', (
                                      '_'.join(str(e) for e in grz_months) + 
                                      "_sub%s" % sbasin))
                if not os.path.exists(outdir):
                    launch_model(grz_months, outdir, herb_csv, grass_csv)
                sum_csv = os.path.join(outdir, 'summary_results.csv')
                sum_df = pandas.read_csv(sum_csv)
                total_gain = sum_df.sum().cow_gain_kg
                sum_biomass = sum_df['sub_%d_calib_dead_kgha' % sbasin] + \
                                sum_df['sub_%d_calib_green_kgha' % sbasin]
                total_biomass = sum_df.sum()['sub_%d_calib_dead_kgha' % sbasin] + \
                               sum_df.sum()['sub_%d_calib_green_kgha' % sbasin]
                mean_biomass = sum_biomass.mean()
                sum_dict['duration'].append(duration)
                sum_dict['grz_months'].append(grz_months)
                sum_dict['gain_kg'].append(total_gain)
                sum_dict['total_biomass'].append(total_biomass)
                sum_dict['avg_biomass'].append(mean_biomass)
                sum_dict['subbasin'].append(sbasin)
        df = pandas.DataFrame(sum_dict)
        df.to_csv(os.path.join( outer_dir, "comparison_8.9.16.csv"), index=False)
    results_dir = os.path.join(outer_dir, 'raw_results')
    erase_intermediate_files(results_dir)
    # full = range(0, 24)
    # herb_csv = "C:/Users/Ginger/Desktop/cow_half_low_sd.csv"
    # outdir = os.path.join(outer_dir, 'raw_results',
                              # '_'.join(str(e) for e in full))
    # launch_model(full, outdir, herb_csv)