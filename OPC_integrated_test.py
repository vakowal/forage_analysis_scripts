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
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_century_link_utils as cent
import forage

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
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\OPC\back_calc_match_last_measurement\match_summary.csv"
    summarize_match(save_as)
    