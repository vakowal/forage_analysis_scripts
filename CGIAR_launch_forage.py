import os
import shutil
import pandas
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def calibrate():
    """Calibrate the model to estimated sustainable capacity of the study area,
    according to the Yauyos Cochas report."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/calibrate_4.27.16/custom_calibration_2"
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
        'outdir': "",
        'template_level': 'GL',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "",
        'grass_csv': "",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
    }
    for anim_type in ['cow', 'sheep', 'camelid']:
        grass_csv = os.path.join(input_dir, 'Pajonal_5_calib.csv')
        herbivore_csv = os.path.join(input_dir, anim_type + '_calib.csv')
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = herbivore_csv
        out_dir = os.path.join(outer_dir, '%s' % anim_type)
        forage_args['outdir'] = out_dir
        # if not os.path.exists(out_dir):
        forage.execute(forage_args)

def launch_biomass_calibration():
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/plant_calibration"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/plant_calibration"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 282,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': "",
        'template_level': 'GL',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "",
        'grass_csv': "",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
        'restart_yearly': 1,
        'diet_verbose': "",
    }
    for subbasin in [4]:
        grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                 subbasin)
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = None
        out_dir = os.path.join(outer_dir, 's%d_run7' % subbasin)
        forage_args['outdir'] = out_dir
        #if not os.path.exists(out_dir):
        forage.execute(forage_args)

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

def launch_baseline():
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/raw_5.2.16"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 204,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': "",
        'template_level': 'GL',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "",
        'grass_csv': "",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
        'restart_yearly': 1,
        'diet_verbose': "",
    }
    for subbasin in range(1, 15):
        grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                 subbasin)
        herbivore_csv = os.path.join(input_dir, 'zero_sd.csv')
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = None
        out_dir = os.path.join(outer_dir, 's%d_zero_sd' % subbasin)
        forage_args['outdir'] = out_dir
        #if not os.path.exists(out_dir):
        forage.execute(forage_args)
                    
def launch_runs():
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/runs_5.31.16"
    for subbasin in [4, 15]:  # range(1, 16):
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sd_level in ['low', 'med', 'high']:
                out_dir = os.path.join(outer_dir, 's%d_%s_%s' %
                                       (subbasin, anim_type, sd_level))
                grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                         subbasin)
                herbivore_csv = os.path.join(input_dir, anim_type + '_' +
                                             sd_level + '_sd.csv')
                if not os.path.exists(out_dir):
                    try:
                        launch_model(input_dir, out_dir, herbivore_csv,
                                     grass_csv)
                    except:
                        continue

def summarize_results():
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\plant_calibration\raw_results"
    summary_dict = {'run': ['0_default'], 
                    'subbasin5': [4454.77], 'subbasin7': [5595.90],
                    'subbasin4': [3119.85]}
    for run in range(1, 16):
        summary_dict['run'].append(run)
        for subbasin in [4, 5, 7]:
            out_dir = os.path.join(outer_dir, 's%d_run%d' % (subbasin, run))
            sum_csv = os.path.join(out_dir, 'summary_results.csv')
            sum_df = pandas.read_csv(sum_csv)
            total_kg_ha = sum_df.iloc[280]['sub_%d_dead_kgha' % subbasin] + \
                          sum_df.iloc[280]['sub_%d_green_kgha' % subbasin]
            summary_dict['subbasin%d' % subbasin].append(total_kg_ha)
    df = pandas.DataFrame(summary_dict)
    save_as = os.path.join(outer_dir, 'run_summary.csv')
    df.to_csv(save_as, index=False)

def launch_model(input_dir, outdir, herb_csv, grass_csv, grz_months=None):
    """Inputs for the forage model in Peru"""
    
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 204,
        'mgmt_threshold': 0.5,
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
        'diet_verbose': 0,
        'restart_yearly': 1,
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

def collect_rotation_results():
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/rotation_high_sd"
    sum_dict = {'duration': [], 'avg_gain_kg': [], 'grz_schedule': [],
                'avg_biomass': [], 'subbasin': [], 'animal_type': [],
                'failed': []}
    total_mos = 204
    sd_level = 'high'
    for anim_type in ['cow', 'sheep', 'camelid']:
        for sbasin in [1, 2, 3, 4, 5, 6, 7, 9]:
            for duration in [1, 2, 3, 4, 6, total_mos]:
                for grz_str in ['p1', 'p2', 'full']:
                    sum_csv = os.path.join(outer_dir, 'raw_results',
                                          '%s_%s_dur%d_%s_sub%s' % (
                                          anim_type, sd_level, duration,
                                          grz_str, sbasin),
                                          'summary_results.csv')
                    if not os.path.exists(sum_csv):
                        continue
                    sum_df = pandas.read_csv(sum_csv)
                    if len(sum_df) < num_months:
                        failed = 1
                    else:
                        failed = 0
                    total_gain = sum_df.sum()['%s_gain_kg' % anim_type]
                    if grz_str == 'full':
                        avg_gain = total_gain / float(total_mos)
                    else:
                        avg_gain = total_gain / (float(total_mos) / 2)
                    sum_biomass = sum_df['sub_%d_dead_kgha' % sbasin] + \
                                    sum_df['sub_%d_green_kgha' % sbasin]
                    mean_biomass = sum_biomass.mean()
                    sum_dict['duration'].append(duration)
                    sum_dict['grz_schedule'].append(grz_str)
                    sum_dict['avg_gain_kg'].append(avg_gain)
                    sum_dict['avg_biomass'].append(mean_biomass)
                    sum_dict['subbasin'].append(sbasin)
                    sum_dict['animal_type'].append(anim_type)
                    sum_dict['failed'].append(failed)
    df = pandas.DataFrame(sum_dict)
    df.to_csv(os.path.join(outer_dir, "comparison_8.17.16.csv"), index=False)

def test_rotation():
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/rotation_high_sd"
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    sum_dict = {'duration': [], 'avg_gain_kg': [], 'grz_months': [],
                'total_biomass': [], 'avg_biomass': [], 'subbasin': []}
    total_mos = 204
    full = range(0, total_mos)
    sd_level = 'high'
    for anim_type in ['cow', 'sheep', 'camelid']:
        herb_csv = os.path.join(input_dir, anim_type + '_' +
                                                 sd_level + '_sd.csv')
        for sbasin in [2, 3, 4, 5, 6, 7, 9]:
            grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' % sbasin)
            for duration in [1, 2, 3, 4, 6]:
                p1, p2 = generate_grz_month_pairs(duration, total_mos)
                for grz_months in [p1, p2, full]:
                    if grz_months == full:
                        duration = total_mos
                    if grz_months == p1:
                        grz_str = 'p1'
                    elif grz_months == p2:
                        grz_str = 'p2'
                    elif grz_months == full:
                        grz_str = 'full'
                    outdir = os.path.join(outer_dir, 'raw_results',
                                          '%s_%s_dur%d_%s_sub%s' % (
                                          anim_type, sd_level, duration,
                                          grz_str, sbasin))
                    if not os.path.exists(outdir):
                        try:
                            launch_model(input_dir, outdir, herb_csv, grass_csv,
                                         grz_months)
                        except:
                            continue
                    erase_intermediate_files(os.path.join(
                                                     outer_dir, 'raw_results'))
                    

def identify_failed_simulations(outer_dir, num_months):
    failed = []
    for folder in os.listdir(outer_dir):
        sum_csv = os.path.join(outer_dir, folder, "summary_results.csv")
        sum_df = pandas.read_csv(sum_csv)
        if len(sum_df) < num_months:
            failed.append(folder)
    print "the following runs failed: "
    print failed

if __name__ == "__main__":
    test_rotation()
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\rotation_high_sd\raw_results"
    num_months = 204
    # identify_failed_simulations(outer_dir, num_months)
    collect_rotation_results()
