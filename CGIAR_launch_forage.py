import os
import shutil
import pandas
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def launch_model(input_dir, outdir, herb_csv, grass_csv):
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
    }
    forage.execute(forage_args)

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
            if file.endswith("summary_results.csv") or file.startswith(
                                                                 "forage-log"):
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
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/rotation_high_sd/raw_results"
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
        'restart_monthly': 0,
    }
    for subbasin in [4]:  # range(1, 15):
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
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/runs_10.18.16"
    for subbasin in [1, 2, 3, 4, 5, 6, 7, 9]:
        for anim_type in ['camelid']:  # ['cow', 'sheep', 'camelid']:
            for sd_level in ['reclow']:  # , 'rechigh']:
                out_dir = os.path.join(outer_dir, 's%d_%s_%s' %
                                       (subbasin, anim_type, sd_level))
                grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                         subbasin)
                herbivore_csv = os.path.join(input_dir, anim_type + '_' +
                                             sd_level + '_sd.csv')
                # if not os.path.exists(out_dir):
                try:
                    launch_model(input_dir, out_dir, herbivore_csv,
                                 grass_csv)
                except:
                    continue

def summarize_results():
    """Summarize results of plant calibration"""
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

def rotation_marginal_value_table():
    """Write marginal value table for input to optimizer from interventions
    including rotational grazing."""
    
    sd_table = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Stocking_density_table.csv"
    sd_df = pandas.read_csv(sd_table)
    marginal_df = {'subbasin': [], 'animal': [], 'density': [],
                   'perc_gain': [], 'total_delta_weight_kg': []}
    time_series_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\biomass_time_series_10.18.16"
    for anim_type in ['cow', 'sheep', 'camelid']:
        for sd_level in ['high', 'low']:  # , 'ighrot', 'lowrot']:
            for sbasin in [4, 15]:  # [1, 2, 3, 4, 5, 6, 7, 9]:               
                sum_csv = os.path.join(time_series_dir, 's%s_%s_%s.csv' % (
                                       sbasin, anim_type, sd_level))
                sum_df = pandas.read_csv(sum_csv)  
                subset = sum_df.loc[sum_df['year'] > 1993]
                grouped = subset.groupby('year')
                avg_yearly_gain = (grouped.sum()[
                                              '%s_gain_kg' % anim_type]).mean()

                if sd_level == 'ighrot' or sd_level == 'high':
                    sd_str = 'rechigh'
                if sd_level == 'lowrot' or sd_level == 'low':
                    sd_str = 'reclow'
                density = sd_df.loc[sd_df['animal_level'] == (
                      '%s_%s' % (anim_type, sd_str))].stocking_density
                start_wt = sum_df.iloc[0]['%s_kg' % anim_type] - \
                               sum_df.iloc[0]['%s_gain_kg' % anim_type]
                avg_yearly_gain_herd = avg_yearly_gain * float(density)
                perc_gain = avg_yearly_gain / float(start_wt)
                marginal_df['subbasin'].append(sbasin)
                marginal_df['animal'].append(anim_type)
                marginal_df['density'].append(sd_level)
                marginal_df['perc_gain'].append(perc_gain)
                marginal_df['total_delta_weight_kg'].append(
                                                  avg_yearly_gain_herd)
    df = pandas.DataFrame(marginal_df)
    marginal_table_path = "C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\marginal_table_10.18.16.csv"
    df.to_csv(marginal_table_path)
                
def collect_rotation_results():
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/rotation_high_sd"
    sum_dict = {'duration': [], 'avg_gain_kg': [], 'grz_schedule': [],
                'avg_biomass': [], 'subbasin': [], 'animal_type': [],
                'failed': [], 'avg_yearly_gain': [], 'sd_level': []}
    total_mos = 204
    for sd_level in ['rechigh', 'reclow']:
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
                        if len(sum_df) < total_mos:
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
                        subset = sum_df.loc[sum_df['year'] > 1993]
                        grouped = subset.groupby('year')
                        avg_yearly_gain = (grouped.sum()[
                                              '%s_gain_kg' % anim_type]).mean()
                        sum_dict['duration'].append(duration)
                        sum_dict['grz_schedule'].append(grz_str)
                        sum_dict['avg_gain_kg'].append(avg_gain)
                        sum_dict['avg_biomass'].append(mean_biomass)
                        sum_dict['subbasin'].append(sbasin)
                        sum_dict['animal_type'].append(anim_type)
                        sum_dict['failed'].append(failed)
                        sum_dict['avg_yearly_gain'].append(avg_yearly_gain)
                        sum_dict['sd_level'].append(sd_level)
    df = pandas.DataFrame(sum_dict)
    df.to_csv(os.path.join(outer_dir, "comparison_8.25.16.csv"), index=False)

def move_summary_files():
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\runs_10.18.16"
    newdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\biomass_time_series_10.18.16"

    if not os.path.exists(newdir):
        os.makedirs(newdir)
    for folder in os.listdir(outerdir):
        elements = folder.split("_")
        summary_file = os.path.join(outerdir, folder,
                                    "summary_results.csv")
        if len(elements) > 3:
            if elements[3] == 'full':
                if elements[1] == 'reclow':
                    elements[1] = 'low'
                elif elements[1] == 'rechigh':
                    elements[1] = 'high'
                new_str = "s%s_%s_%s" % (elements[4][-1:], elements[0],
                                         elements[1])
        else:
            if elements[2].startswith('rec'):
                elements[2] = elements[2][3:]
            if elements[0].startswith('s'):
                elements[0] = elements[0][1:]
            new_str = "s%s_%s_%s" % (elements[0], elements[1], elements[2])
        new_path = os.path.join(newdir, new_str + ".csv")
        if os.path.isfile(new_path):
            continue
        else:
            try:
                shutil.copyfile(summary_file, new_path)
            except IOError:
                continue

def calc_SWAT_inputs():
    """Calculate input management parameters for SWAT from biomass time
    series."""
    
    biomass_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\biomass_time_series_8.25.16"
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results"
    
    sum_dict = {'subbasin': [], 'animal': [], 'intensity': [],
                'biomass_consumed': [], 'manure_kg': []}
    for sbasin in [1, 2, 3, 4, 5, 6, 7, 9]:
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sd_level in ['low', 'high']:
                filename = 's%s_%s_%s.csv' % (sbasin, anim_type, sd_level)
                sum_df = pandas.read_csv(os.path.join(biomass_dir, filename))
                grouped = sum_df.groupby('year')
                avg_yearly_offtake = (grouped.sum()['total_offtake']).mean()
                manure_kg = 2.27 * 0.2 * (avg_yearly_offtake / 2.5)
                sum_dict['subbasin'].append(sbasin)
                sum_dict['animal'].append(anim_type[:3])
                sum_dict['intensity'].append(sd_level[-3:])
                sum_dict['biomass_consumed'].append(avg_yearly_offtake)
                sum_dict['manure_kg'].append(manure_kg)
    df = pandas.DataFrame(sum_dict)
    df = df[['subbasin', 'animal', 'intensity', 'biomass_consumed',
             'manure_kg']]
    df.to_csv(os.path.join(out_dir, "SWAT_inputs_8.29.16.csv"), index=False)            

def calculate_rotated_time_series():
    """Make biomass time series for Perrine by averaging two rotated summary
    results files, those pertaining to different rotation schedules but with
    otherwise identical inputs."""
    
    time_series_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\biomass_time_series_10.18.16"
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\rotation_10.18.16"
    total_mos = 204
    duration = 4
    for sd_level in ['rechigh', 'low']:
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sbasin in [1, 2, 3, 4, 5, 6, 7, 9]:
                p1_csv = os.path.join(outer_dir, 'raw_results',
                                      '%s_%s_dur%d_p1_sub%s' % (
                                      anim_type, sd_level, duration, sbasin),
                                      'summary_results.csv')
                p2_csv = os.path.join(outer_dir, 'raw_results',
                                      '%s_%s_dur%d_p2_sub%s' % (
                                      anim_type, sd_level, duration, sbasin),
                                      'summary_results.csv')
                if not os.path.exists(p1_csv) or not os.path.exists(p2_csv):
                    import pdb; pdb.set_trace()
                sum_p1 = pandas.read_csv(p1_csv, index_col=0)
                sum_p2 = pandas.read_csv(p2_csv, index_col=0)
                if len(sum_p1) < total_mos or len(sum_p2) < total_mos:
                    raise Exception
                ### calculate average
                ave_df = sum_p1.copy()
                for ci in [0, 1, 2, 3, 6, 7, 8]:
                    col1 = sum_p1.ix[:, ci]
                    col2 = sum_p2.ix[:, ci]
                    comb = pandas.concat([col1, col2], axis=1)
                    comb_mean = comb.mean(axis=1)
                    ave_df.ix[:, ci] = comb_mean
                sd_str = sd_level[-3:] + 'rot'
                filename = 's%s_%s_%s.csv' % (sbasin, anim_type, sd_str)
                save_as = os.path.join(time_series_dir, filename)
                ave_df.to_csv(save_as)
    
def test_rotation():
    """Run the model for Peru, including two complementary rotation schedules
    and a "full" (non-rotated) grazing schedule."""
    
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 204,
        'mgmt_threshold': 0.2,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': '',
        'template_level': 'GH',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': '',
        'grass_csv': '',
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': '',
        'diet_verbose': 0,
        'restart_yearly': 1,
        'restart_monthly': 1,
        'grz_months': '',
    }
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/rotation_10.18.16"
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    sum_dict = {'duration': [], 'avg_gain_kg': [], 'grz_months': [],
                'total_biomass': [], 'avg_biomass': [], 'subbasin': []}
    total_mos = 204
    full = range(0, total_mos)
    duration = 4
    p1, p2 = generate_grz_month_pairs(duration, total_mos)
    for anim_type in ['camelid']:  # ['cow', 'sheep', 'camelid']:
        for sbasin in [1, 2, 3, 4, 5, 6, 7, 9]:
            grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' % sbasin)
            for sd_level in ['reclow']:  # , 'rechigh']:
                herb_csv = os.path.join(input_dir, anim_type + '_' +
                                                 sd_level + '_sd.csv')
                for grz_months in [p1, p2]:  #, full]:
                    if grz_months == full:
                        dur = total_mos
                    else:
                        dur = duration
                    if grz_months == p1:
                        grz_str = 'p1'
                    elif grz_months == p2:
                        grz_str = 'p2'
                    elif grz_months == full:
                        grz_str = 'full'
                    outdir = os.path.join(outer_dir, 'raw_results',
                                          '%s_%s_dur%d_%s_sub%s' % (
                                          anim_type, sd_level, dur,
                                          grz_str, sbasin))
                    if not os.path.exists(outdir):
                    # sum_csv = os.path.join(outdir, "summary_results.csv")
                    # sum_df = pandas.read_csv(sum_csv)
                    # if len(sum_df) < total_mos:
                        # forage_args['mgmt_threshold'] = 0.3
                        forage_args['outdir'] = outdir
                        forage_args['herbivore_csv'] = herb_csv
                        forage_args['grass_csv'] = grass_csv
                        forage_args['input_dir'] = input_dir
                        forage_args['grz_months'] = grz_months
                        try:
                            forage.execute(forage_args)
                        except:
                            continue
                erase_intermediate_files(os.path.join(
                                                 outer_dir, 'raw_results'))
                    

def identify_failed_simulations(outer_dir, num_months):
    failed = []
    for folder in os.listdir(outer_dir):
        sum_csv = os.path.join(outer_dir, folder, "summary_results.csv")
        if not os.path.exists(sum_csv):
            failed.append(folder)
            continue
        sum_df = pandas.read_csv(sum_csv)
        if len(sum_df) < num_months:
            failed.append(folder)
    print "the following runs failed: "
    print failed

if __name__ == "__main__":
    # test_rotation()
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\runs_10.18.16"
    num_months = 204
    test_rotation()
    # identify_failed_simulations(outer_dir, num_months)
    # collect_rotation_results()
    # identify_failed_simulations(outer_dir, num_months)
    # calculate_rotated_time_series()
    launch_runs()
    # move_summary_files()
    # rotation_marginal_value_table()
    # calc_SWAT_inputs()
    