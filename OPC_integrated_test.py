## integrated test of forage model on OPC

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas
import numpy as np
import back_calculate_management as backcalc
sys.path.append(
 'C:/Users/Ginger/Documents/Python/rangeland_production')
import forage_century_link_utils as cent
import forage

def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""
    
    df = pandas.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)

def calculate_new_growth(run_dir, grass_csv, year):
    """Calculate the percentage of biomass that is new growth each month.
    run_dir is the directory of the model run. grass_csv is the input csv
    describing grasses used in the simulation. year is the year for which we
    will calculate percent new growth."""
    
    cent_dir = "CENTURY_outputs_m12_y%s" % str(year)
    grass_list = pandas.read_csv(grass_csv)
    n_mult = grass_list.get_value(0, "N_multiplier")
    gr = grass_list.label.unique()
    assert len(gr) == 1, "We can only handle one grass type"
    gr = gr[0]
    filename = os.path.join(run_dir, cent_dir, "%s.lis" % gr)
    gr_df = pandas.read_fwf(filename)
    sub_df = gr_df.loc[gr_df['time'] > year]
    sub_df = sub_df.loc[sub_df['time'] <= year + 1]
    sub_df = sub_df.drop_duplicates('time')
    growth = sub_df.agcacc.values * 2.5
    new_growth = [growth[0]]
    for idx in range(1, 12):
        new_growth.append(growth[idx] - growth[idx - 1])
    live_biomass = sub_df.aglivc.values * 2.5
    stded_biomass = sub_df.stdedc.values * 2.5
    biomass = live_biomass + stded_biomass
    perc_green = (live_biomass / biomass).tolist()
    n_content_live = (sub_df['aglive(1)'].values) * 6.25 / live_biomass
    n_content_live = np.multiply(n_content_live, n_mult).tolist()
    n_content_dead = (sub_df['stdede(1)'].values) * 6.25 / stded_biomass
    n_content_dead = np.multiply(n_content_dead, n_mult).tolist()
    
    live_weighted = [a * b for a, b in zip(live_biomass, n_content_live)]
    dead_weighted = [a * b for a, b in zip(stded_biomass, n_content_dead)]
    weighted_sum = [a + b for a, b in zip(live_weighted, dead_weighted)]
    weighted_cp_avg = [a / b for a, b in zip(weighted_sum, biomass)]
    results_list = [new_growth, live_biomass, biomass, n_content_live,
                    weighted_cp_avg, perc_green]
    return results_list

def calc_n_mult(forage_args, target):
    """Calculate N multiplier for a grass to achieve target crude protein
    content, and edit grass input to include that N multiplier. Target reflects
    crude protein of live grass. Target should be supplied as a float between 0
    and 1."""
    
    tolerance = 0.001  # must be within this proportion of target value
    grass_df = pandas.read_csv(forage_args['grass_csv'])
    grass_label = grass_df.iloc[0].label
    # args copy to launch model to calculate n_mult
    args_copy = forage_args.copy()
    args_copy['outdir'] = os.path.join(os.path.dirname(forage_args['outdir']),
                                       '{}_n_mult_calc'.format(
                                      os.path.basename(forage_args['outdir'])))
    # find correct output time period
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
        year = ((forage_args['num_months'] - 1) / 12) + forage_args[u'start_year']
    intermediate_dir = os.path.join(args_copy['outdir'],
                                    'CENTURY_outputs_m%d_y%d' % (month, year))
    sim_output = os.path.join(intermediate_dir, '{}.lis'.format(grass_label))
    first_year = forage_args['start_year']
    last_year = year
    
    def get_raw_cp_green():
        # calculate n multiplier to achieve target
        outputs = cent.read_CENTURY_outputs(sim_output, first_year, last_year)
        outputs.drop_duplicates(inplace=True)
        
        # restrict to months of the simulation
        first_month = forage_args[u'start_month']
        start_date = first_year + float('%.2f' % (first_month / 12.))
        end_date = last_year + float('%.2f' % (month / 12.))
        outputs = outputs[(outputs.index >= start_date)]
        outputs = outputs[(outputs.index <= end_date)]
        return np.mean(outputs.aglive1 / outputs.aglivc)
    
    def set_n_mult():
        # edit grass csv to reflect calculated n_mult
        grass_df = pandas.read_csv(forage_args['grass_csv'])
        grass_df.N_multiplier = grass_df.N_multiplier.astype(float)
        grass_df = grass_df.set_value(0, 'N_multiplier', float(n_mult))
        grass_df = grass_df.set_index('label')
        grass_df.to_csv(forage_args['grass_csv'])
    
    n_mult = 1
    set_n_mult()
    forage.execute(args_copy)
    cp_green = get_raw_cp_green()
    diff = abs(target - (n_mult * cp_green))
    while diff > tolerance:
        n_mult = '%.10f' % (target / cp_green)
        set_n_mult()
        forage.execute(args_copy)
        cp_green = get_raw_cp_green()
        diff = abs(target - (float(n_mult) * cp_green))

def stocking_density_percent_new_growth_test():
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\stocking_density_new_growth\n_mult_start_2014\growth_summary.csv"
    grass_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\zero_dens_2013\OPC_avg.csv"
    herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\herd_avg_uncalibrated.csv"
    outer_out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\stocking_density_new_growth\n_mult_start_2014"
    target = 0.14734
    sum_dict = {'stocking_density': [], 'year': [], 'month': [], 'label': [],
                'biomass': []}
    n_mult_dict = {'stocking_density': [], 'n_mult': []}
    forage_args = {
        'latitude': -0.0040,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 2014,
        'start_month': 11,
        'num_months': 14,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'template_level': 'GL',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'herbivore_csv': herb_csv,
        'grass_csv': grass_csv,
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\zero_dens",
        'restart_yearly': 0,
        'diet_verbose': 0,
        'restart_monthly': 1,
    }
    for sd in [0.1, 0.45, 0.8]:
        n_mult_dict['stocking_density'].append(sd)
        modify_stocking_density(herb_csv, sd)
        outdir = os.path.join(outer_out_dir, 'cattle_%s' % str(sd))
        forage_args['outdir'] = outdir
        calc_n_mult(forage_args, target)
        grass_df = pandas.read_csv(grass_csv)
        n_mult = grass_df.iloc[0].N_multiplier
        n_mult_dict['n_mult'].append(n_mult)
        forage.execute(forage_args)
        for year in [2014, 2015]:   
            sum_dict['year'] = sum_dict['year'] + [year] * 96
            sum_dict['month'] = sum_dict['month'] + (range(1, 13) * 8)
            sum_dict['stocking_density'] = sum_dict['stocking_density'] + [sd] * 96
            sum_dict['label'] = sum_dict['label'] + ['new_growth'] * 12 + \
                                                    ['live_biomass'] * 12 + \
                                                    ['total_biomass'] * 12 + \
                                                    ['n_content_live'] * 12 + \
                                                    ['weighted_cp_avg'] * 12 + \
                                                    ['perc_green'] * 12 + \
                                                    ['liveweight_gain'] * 12 + \
                                                    ['liveweight_gain_herd'] * 12
            new_growth_results_list = calculate_new_growth(outdir, grass_csv, year)
            new_growth = new_growth_results_list[0]
            live_biomass = new_growth_results_list[1].tolist()
            biomass = new_growth_results_list[2].tolist()
            n_content_live = new_growth_results_list[3]
            weighted_cp_avg = new_growth_results_list[4]
            perc_green = new_growth_results_list[5]
            n_content = np.multiply(n_content_live, n_mult).tolist()
            weight_df = pandas.read_csv(os.path.join(outdir, 'summary_results.csv'))
            weight_df = weight_df.loc[weight_df['year'] == year]
            gain = weight_df.cattle_gain_kg.values.tolist()
            if len(gain) < 12:
                gain = [0] * (12 - len(gain)) + gain
            gain_herd = np.multiply(gain, sd).tolist()
            sum_dict['biomass'] = sum_dict['biomass'] + new_growth + \
                                    live_biomass + \
                                    biomass + n_content_live + \
                                    weighted_cp_avg + perc_green + gain + \
                                    gain_herd
    try:
        sum_df = pandas.DataFrame(sum_dict)
        sum_df.to_csv(save_as, index=False)
        n_mult_df = pandas.DataFrame(n_mult_dict)
        n_mult_df.to_csv(r"C:\Users\Ginger\Desktop\n_mult.csv")
    except:
        import pdb; pdb.set_trace()
    

def back_calc_match_last_meas():
    """Use the back-calc management routine to match the last empirical biomass
    measurement within 2 km of a weather station on OPC.  These biomass
    measurements are taken from my summary derived from the data
    OPC_veg_9.30.16."""
    
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    n_months = 24
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC_integrated_test\back_calc_match_last_measurement"
    vary = 'both' # 'schedule', 'intensity', 'both'
    live_or_total = 'total'  # 'live' (live biomass) or 'total' (live + standing dead biomass)
    threshold = 10.0  # must match biomass within this many g per sq m
    max_iterations = 40  # number of times to try
    fix_file = 'drytrpfi.100'
    template_level = 'GLP'

    site_list = sites_definition('last')
    for site in site_list:
        out_dir_site = os.path.join(out_dir, site['name'])
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site) 
            backcalc.back_calculate_management(site, input_dir, century_dir, out_dir_site,
                                      fix_file, n_months, vary, live_or_total,
                                      threshold, max_iterations, template_level)

def back_calc_match_first_meas():
    """Use the back-calc management routine to match the first empirical biomass
    measurement within 2 km of a weather station on OPC.  These biomass
    measurements are taken from my summary derived from the data
    OPC_veg_9.30.16."""
    
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    n_months = 24
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC_integrated_test\back_calc_match_first_measurement"
    vary = 'both' # 'schedule', 'intensity', 'both'
    live_or_total = 'total'  # 'live' (live biomass) or 'total' (live + standing dead biomass)
    threshold = 10.0  # must match biomass within this many g per sq m
    max_iterations = 40  # number of times to try
    fix_file = 'drytrpfi.100'
    
    site_list = sites_definition('first')
    for site in site_list:
        out_dir_site = os.path.join(out_dir, site['name'])
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site) 
            backcalc.back_calculate_management(site, input_dir, century_dir, out_dir_site,
                                      fix_file, n_months, vary, live_or_total,
                                      threshold, max_iterations)

def collect_results(outerdir, site_list):
    succeeded = []
    failed = []
    diff_list = []

    for site in site_list:
        date = site['date']
        site_name = site['name']
        result_csv = os.path.join(outerdir, site_name,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pandas.read_csv(result_csv)
        diff = res_df.iloc[len(res_df) - 1].Simulated_biomass - \
                           res_df.iloc[len(res_df) - 1].Empirical_biomass
        diff_list.append(diff)
        if abs(float(diff)) <= 15.0:
            succeeded.append(site_name)
        else:
            failed.append(site_name)
    print "these sites succeeded: {}".format(succeeded)
    print "these sites failed: {}".format(failed)
    print "diff list:"
    print diff_list

def combine_empirical_density_files(save_as):
    outerdir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\From_Sharon_5.29.15\Matched_GPS_records\Matched_with_weather_stations'
    files = [f for f in os.listdir(outerdir) if
             f.startswith('average_animals')]
    df_list = list()
    for f in files:
        site_name = re.search('average_animals_(.+?)_2km.csv', f).group(1)
        csv = os.path.join(outerdir, f)
        df = pandas.read_csv(csv)
        df['site'] = site_name
        df_list.append(df)
    combined_df = pandas.concat(df_list)
    combined_df.to_csv(save_as)

def sites_definition(first_or_last):
    if first_or_last == 'first':
        kamok = {'name': 'Kamok', 'biomass': 119.72, 'date': 2015.17}
        loidien = {'name': 'Loidien', 'biomass': 164.74, 'date': 2015.17}
        research = {'name': 'Research', 'biomass': 271.7, 'date': 2015.17}
        loirugurugu = {'name': 'Loirugu', 'biomass': 105.97, 'date': 2015.25}
        serat = {'name': 'Serat', 'biomass': 100.08, 'date': 2015.33}
        rongai = {'name': 'Rongai', 'biomass': 175.92, 'date': 2015.33}
    elif first_or_last == 'last':
        golf7 = {'name': 'Golf_7', 'biomass': 152.35, 'date': 2015.33}
        sirima = {'name': 'Sirima', 'biomass': 190.42, 'date': 2016.00}
        kamok = {'name': 'Kamok', 'biomass': 119.72, 'date': 2015.92}
        loidien = {'name': 'Loidien', 'biomass': 93.74, 'date': 2015.92}
        research = {'name': 'Research', 'biomass': 276.03, 'date': 2016.00}
        loirugurugu = {'name': 'Loirugu', 'biomass': 96.76, 'date': 2015.75}
        serat = {'name': 'Serat', 'biomass': 145.7, 'date': 2016.00}
        rongai = {'name': 'Rongai', 'biomass': 165.64, 'date': 2016.00}
    site_list = [golf7, sirima, kamok, loidien, research, loirugurugu, serat,
                 rongai]
    return site_list

def empirical_series_definition():
    """Lazy me, I typed these up manually rather than reading them from the csv
    that summarizes empirical stocking density series calculated from GPS
    records ("NatCap_backup\Forage_model\Data\Kenya\From_Sharon\
    From_Sharon_5.29.15\Matched_GPS_records\Matched_with_weather_stations\
    sites_combined.csv").  grz_months gives months where grazing took place
    (where 0=January 2015) and dens_series gives stocking density in that
    month, in animals per ha."""
    
    kamok = {'name': 'Kamok', 'grz_months': [1, 2, 3, 8], 'dens_series':
             {1: 0.031, 2: 0.029, 3: 0.015, 8: 0.007}, 'start_mo': 2}
    loidien = {'name': 'Loidien', 'grz_months': [1, 2], 'dens_series':
               {1: 0.011, 2: 0.012}, 'start_mo': 2}
    research = {'name': 'Research', 'grz_months': [1, 2, 3, 4, 6, 7, 9],
                'dens_series': {1: 0.003, 2: 0.013, 3: 0.016, 4: 0.022,
                6: 0.001, 7: 0.005, 9: 0.003}, 'start_mo': 2}
    loirugurugu = {'name': 'Loirugu', 'grz_months': [5], 'dens_series':
                   {5: 0.009}, 'start_mo': 3}
    serat = {'name': 'Serat', 'grz_months': [6, 7], 'dens_series': {6: 0.003,
             7: 0.004}, 'start_mo': 4}
    rongai = {'name': 'Rongai', 'grz_months': [], 'dens_series': {},
              'start_mo': 4}
    site_list = [kamok, loidien, research, loirugurugu, serat, rongai]
    for site in site_list:
        site['dens_series'] = {k: v * 10 for k, v in
                               site['dens_series'].items()}
    return site_list

def run_zero_density():
    """Compare productivity at the 6 sites with zero grazing pressure."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/zero_dens"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\zero_dens"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = empirical_series_definition()
    for site in site_list:
        outdir = os.path.join(outer_output_dir, '{}'.format(site['name']))
        grass_csv = os.path.join(input_dir, '{}.csv'.format(site['name']))
        forage_args = {
            'latitude': 0.02759,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2015,
            'start_month': 1,
            'num_months': 12,
            'mgmt_threshold': 0.1,
            'century_dir': century_dir,
            'outdir': outdir,
            'template_level': 'GL',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 0,
            'user_define_digestibility': 0,
            'herbivore_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herd_avg_calibrated.csv",
            'grass_csv': grass_csv,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            'input_dir': input_dir,
            'diet_verbose': 0,
            'restart_monthly': 1,
            'restart_yearly': 0,
            'grz_months': [],
            }
        forage.execute(forage_args)
    
def run_empirical_stocking_density():
    """Run the model forward from the date of the first empirical biomass
    measurement, using the back-calculated management history that was
    calculated to match the first measurement.  Impose empirical stocking
    density as calculated from GPS records."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/integrated_test"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\empirical_stocking_density"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = empirical_series_definition()
    for site in site_list:
        start_mo = site['start_mo']
        # find graz file associated with back-calc management
        graz_filter = [f for f in os.listdir(input_dir) if
                       f.startswith('graz_{}'.format(site['name']))]
        if len(graz_filter) == 1:
            graz_file = os.path.join(input_dir, graz_filter[0])
            def_graz_file = os.path.join(century_dir, 'graz.100')
            shutil.copyfile(def_graz_file, os.path.join(century_dir,
                            'default_graz.100'))
            shutil.copyfile(graz_file, def_graz_file)
        outdir = os.path.join(outer_output_dir, site['name'])
        grass_csv = os.path.join(input_dir, '{}.csv'.format(site['name']))
        forage_args = {
            'latitude': 0.02759,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2015,
            'start_month': start_mo,
            'num_months': 13 - start_mo,
            'mgmt_threshold': 0.1,
            'century_dir': century_dir,
            'outdir': outdir,
            'template_level': 'GL',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 0,
            'user_define_digestibility': 0,
            'herbivore_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herd_avg_calibrated.csv",
            'grass_csv': grass_csv,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            'input_dir': input_dir,
            'diet_verbose': 0,
            'restart_monthly': 1,
            'restart_yearly': 0,
            'grz_months': site['grz_months'],
            'density_series': site['dens_series'],
            }
        forage.execute(forage_args)
        os.remove(def_graz_file)
        shutil.copyfile(os.path.join(century_dir, 'default_graz.100'),
                        def_graz_file)
        os.remove(os.path.join(century_dir, 'default_graz.100'))

def summarize_comparison(save_dir):
    """Compare biomass calculated via empirical stocking density test to
    biomass measurements from the field."""
    
    sim_outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\empirical_stocking_density"
    emp_list = [r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\OPC_veg_9.30.16_by_weather.csv"]
                # r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\OPC_veg_9.30.16_by_weather_unrestricted_by_trees_shrubs.csv"]
    for empirical_csv in emp_list:
        emp_df = pandas.read_csv(empirical_csv)
        kamok = {'name': 'Kamok'}
        loidien = {'name': 'Loidien'}
        research = {'name': 'Research'}
        loirugurugu = {'name': 'Loirugurugu'}
        serat = {'name': 'Serat'}
        rongai = {'name': 'Rongai'}
        site_list = [kamok, loidien, research, loirugurugu, serat, rongai]
        comp_dict = {'site': [], 'month': [], 'biomass_kg_ha': [],
                     'sim_vs_emp': []}
        for site in site_list:
            emp_sub = emp_df.ix[(emp_df['site'] == site['name']) &
                                (emp_df['year'] == 15), ['mean_biomass_kgha',
                                'month']]
            match_months = emp_sub['month'].values.tolist()
            sim_dir = os.path.join(sim_outerdir, '{}_x10'.format(site['name']))
            template_dir = os.path.join(sim_dir, 'CENTURY_outputs_spin_up')
            template_file = [f for f in os.listdir(template_dir) if
                             f.endswith('.bin')][0]
            sim_name = template_file[:-4]
            sim_df = pandas.read_csv(os.path.join(sim_dir,
                                                  'summary_results.csv'))
            sim_df['total_kgha'] = sim_df['{}_green_kgha'.format(sim_name)] + \
                                        sim_df['{}_dead_kgha'.format(sim_name)]
            sim_df = sim_df.loc[(sim_df['month'].isin(match_months)),
                                ['total_kgha', 'month']]
            comp_dict['site'].extend([site['name']] * len(match_months) * 2)
            comp_dict['month'].extend(sim_df.month.values)
            comp_dict['month'].extend(emp_sub.month.values)
            comp_dict['sim_vs_emp'].extend(['sim'] * len(match_months))
            comp_dict['sim_vs_emp'].extend(['emp'] * len(match_months))
            comp_dict['biomass_kg_ha'].extend(sim_df.total_kgha.values)
            comp_dict['biomass_kg_ha'].extend(emp_sub.mean_biomass_kgha.values)
        df = pandas.DataFrame(comp_dict)
        save_as = os.path.join(save_dir, 'comparison_x10_{}'.format(
                                              os.path.basename(empirical_csv)))
        df.to_csv(save_as)

def combine_summary_files():
    """Make a file that can be used to plot biomass differences between
    sites."""
    
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\zero_dens\combined_summary.csv"
    df_list = []
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\zero_dens"
    site_list = empirical_series_definition()
    for site in site_list:
        sim_name = site['name']
        sim_dir = os.path.join(outer_output_dir, '{}'.format(sim_name))
        sim_df = pandas.read_csv(os.path.join(sim_dir,'summary_results.csv'))
        sim_df['total_kgha'] = sim_df['{}_green_kgha'.format(sim_name)] + \
                                    sim_df['{}_dead_kgha'.format(sim_name)]
        sim_df['site'] = sim_name
        sim_df = sim_df[['site', 'total_kgha', 'month', 'year',
                         'total_offtake']]
        df_list.append(sim_df)
    combined_df = pandas.concat(df_list)
    combined_df.to_csv(save_as)

def summarize_sch_wrapper():
    """summarize back-calculated schedules in terms of biomass removed"""
    
    n_months = 12
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC_integrated_test\back_calc_match_last_measurement"
    site_list = sites_definition('last')
    raw_file = os.path.join(outerdir, 'schedule_summary.csv')
    summary_file = os.path.join(outerdir, 'percent_removed.csv')
    backcalc.summarize_calc_schedules(site_list, n_months, input_dir,
                                      century_dir, outerdir, raw_file,
                                      summary_file)
                                              
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC_integrated_test\back_calc_match_first_measurement"
    # site_list = sites_definition('first')
    raw_file = os.path.join(outerdir, 'schedule_summary.csv')
    summary_file = os.path.join(outerdir, 'percent_removed.csv')
    # backcalc.summarize_calc_schedules(site_list, n_months, input_dir, 
                                      # century_dir, outerdir, raw_file,
                                      # summary_file)

def summarize_live_dead(save_as, input_dir, n_months):
    """summarize live:dead ratio in the last n_months months of a schedule
    estimated by back-calc management routine."""
    
    sum_dict = {'site': [], 'avg_ratio_live_dead': [], 'avg_live': [],
                'avg_dead': []}
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\back_calc_match_last_measurement"
    site_list = sites_definition('last')
    for site in site_list:
        site_name = site['name']
        empirical_date = site['date']
        site_dir = os.path.join(out_dir, site_name)
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [int(re.search('{}_{}(.+?).sch'.format(site_name,
                         site_name), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:  # no schedule modification was needed
            final_sch = os.path.join(input_dir, '{}.sch'.format(site_name))
        else:
            final_sch_iter = max(sch_iter_list)
            final_sch = os.path.join(site_dir, '{}_{}{}.sch'.format(site_name,
                                     site_name, final_sch_iter))
        
        # read schedule file to know
        schedule_df = cent.read_block_schedule(final_sch)
        for i in range(0, schedule_df.shape[0]):
            start_year = schedule_df.loc[i, 'block_start_year']
            last_year = schedule_df.loc[i, 'block_end_year']
            if empirical_date > start_year and empirical_date <= last_year:
                break
        relative_empirical_year = int(math.floor(empirical_date) -
                                      start_year + 1)
        empirical_month = int(round((empirical_date - float(math.floor(
                                empirical_date))) * 12))
        if empirical_month == 0:
            empirical_month = 12
            relative_empirical_year = relative_empirical_year - 1
        first_rel_month, first_rel_year = cent.find_first_month_and_year(
                                               n_months, empirical_month,
                                               relative_empirical_year)
        first_abs_year = first_rel_year + start_year - 1
        
        
        result_csv = os.path.join(site_dir,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pandas.read_csv(result_csv)
        final_sim = int(res_df.iloc[len(res_df) - 1].Iteration)
        output_file = os.path.join(site_dir,
                               'CENTURY_outputs_iteration{}'.format(final_sim),
                               '{}.lis'.format(site_name))
        biomass_df = cent.read_CENTURY_outputs(output_file, first_abs_year,
                                               math.floor(empirical_date))
        biomass_df['time'] = biomass_df.index
        biomass_df = biomass_df.loc[biomass_df['time'] <= empirical_date]
        biomass_df = biomass_df.iloc[-24:]
        avg_live_dead = (biomass_df.aglivc / biomass_df.stdedc).mean()
        avg_live = biomass_df.aglivc.mean()
        agv_dead = biomass_df.stdedc.mean()
        sum_dict['site'].append(site_name)
        sum_dict['avg_ratio_live_dead'].append(avg_live_dead)
        sum_dict['avg_live'].append(avg_live)
        sum_dict['avg_dead'].append(agv_dead)
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as)
        
def summarize_match(save_as):
    """summarize the results of back-calc management routine by collecting
    the final comparison of empirical vs simulated biomass from each site
    where the routine was run."""
    
    sum_dict = {'site': [], 'g_m2': [], 'sim_vs_emp': []}
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\back_calc_match_last_measurement"
    site_list = sites_definition('last')
    for site in site_list:
        site_name = site['name']
        site_dir = os.path.join(out_dir, site_name)
        result_csv = os.path.join(site_dir,
                                  'modify_management_summary_{}.csv'.
                                  format(site_name))
        res_df = pandas.read_csv(result_csv)
        sum_dict['site'].extend([site_name] * 2)
        sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                Simulated_biomass)
        sum_dict['sim_vs_emp'].append('sim')
        sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                Empirical_biomass)
        sum_dict['sim_vs_emp'].append('emp')
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as)
    
if __name__ == "__main__":
    # run_empirical_stocking_density()
    # save_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\empirical_stocking_density"
    # summarize_comparison(save_dir)
    # combine_summary_files()
    # back_calc_match_first_meas()
    # back_calc_match_last_meas()
    # summarize_sch_wrapper()
    # save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\back_calc_match_last_measurement\live_dead_summary.csv"
    # summarize_match(save_as)
    # n_months = 24
    # input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    # summarize_live_dead(save_as, input_dir, n_months)
    stocking_density_percent_new_growth_test()
    