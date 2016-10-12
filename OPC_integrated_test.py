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
    n_years = 2 # how many years to potentially manipulate?
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\back_calc_match_last_measurement"
    vary = 'both' # 'schedule', 'intensity', 'both'
    live_or_total = 'total'  # 'live' (live biomass) or 'total' (live + standing dead biomass)
    threshold = 10.0  # must match biomass within this many g per sq m
    max_iterations = 40  # number of times to try
    fix_file = 'drytrpfi.100'

    kamok = {'name': 'Kamok', 'biomass': 119.72, 'date': 2015.92}
    loidien = {'name': 'Loidien', 'biomass': 93.74, 'date': 2015.92}
    research = {'name': 'Research', 'biomass': 276.03, 'date': 2016.00}
    loirugurugu = {'name': 'Loirugu', 'biomass': 96.76, 'date': 2015.75}
    serat = {'name': 'Serat', 'biomass': 145.7, 'date': 2016.00}
    rongai = {'name': 'Rongai', 'biomass': 165.64, 'date': 2016.00}


    site_list = [kamok, loidien, research, loirugurugu, serat, rongai]
    for site in site_list:
        out_dir_site = os.path.join(out_dir, site['name'])
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site) 
        backcalc.back_calculate_management(site, input_dir, century_dir, out_dir_site,
                                  fix_file, n_years, vary, live_or_total,
                                  threshold, max_iterations)

def back_calc_match_first_meas():
    """Use the back-calc management routine to match the first empirical biomass
    measurement within 2 km of a weather station on OPC.  These biomass
    measurements are taken from my summary derived from the data
    OPC_veg_9.30.16."""
    
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    n_years = 2 # how many years to potentially manipulate?
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\back_calc_match_first_measurement"
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
                                  fix_file, n_years, vary, live_or_total,
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

def summarize_calc_schedules(site_list, outerdir, save_as):
    """summarize the grazing schedules that were calculated via the back-calc
    management regime.
    We summarize grazing scheduled only during 2015, and we assume that 2015 is
    relative year 3 within the last block of the schedule file."""
    
    relative_year_2015 = 3
    df_list = list()
    for site in site_list:
        flgrem_GLP = 0.1
        fdgrem_GLP = 0.05
        flgrem_GL = 0.1
        fdgrem_GL = 0.05
        site_name = site['name']
        empirical_date = site['date']
        site_dir = os.path.join(outerdir, site_name)
        result_csv = os.path.join(site_dir,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pandas.read_csv(result_csv)
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [int(re.search('{}_{}(.+?).sch'.format(site_name,
                         site_name), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:
            import pdb; pdb.set_trace()
        final_sch_iter = max(sch_iter_list)
        final_sch = os.path.join(site_dir, '{}_{}{}.sch'.format(site_name,
                                 site_name, final_sch_iter))
        
        # read schedule file, collect months where grazing was scheduled
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
        
        # find months where grazing took place prior to empirical date
        graz_schedule = cent.read_graz_level(final_sch)
        block = graz_schedule.loc[(graz_schedule["block_end_year"] == 
                                  last_year), ['relative_year', 'month',
                                  'grazing_level']]
        empirical_year = block.loc[(block['relative_year'] ==
                                   relative_empirical_year), ]
        empirical_year = empirical_year.loc[(empirical_year['month'] <=
                                            empirical_month), ]
        prev_year = block.loc[(block['relative_year'] <
                              relative_empirical_year), ]
        prev_year = prev_year.loc[(prev_year['relative_year'] >=
                                   relative_year_2015), ]
        history = prev_year.append(empirical_year)
        if len(history) > 0:
            # collect % biomass removed for these months
            grz_files = [f for f in os.listdir(site_dir) if
                         f.startswith('graz')]
            if len(grz_files) > 0:
                grz_iter_list = [int(re.search('graz_{}(.+?).100'.format(
                                 site_name), f).group(1)) for f in grz_files]
                final_iter = max(grz_iter_list)
                final_grz = os.path.join(site_dir, 'graz_{}{}.100'.format(
                                         site_name, final_iter))
                with open(final_grz, 'rb') as old_file:
                    for line in old_file:
                        if 'GL    ' in line:
                            line = old_file.next()
                            if 'FLGREM' in line:
                                flgrem_GL = line[:8].strip()
                                line = old_file.next()
                            if 'FDGREM' in line:
                                fdgrem_GL = line[:8].strip()
                            else:
                                er = "Error: FLGREM expected"
                                raise Exception(er)
        history['perc_live_removed'] = np.where(history['grazing_level']
                                                == 'GL', flgrem_GL, flgrem_GLP)
        history['perc_dead_removed'] = np.where(history['grazing_level']
                                                == 'GL', fdgrem_GL, fdgrem_GLP)
        # collect biomass for these months
        history['year'] = np.where(history['relative_year'] == 3, 2015, 2016)
        history['date'] = history.year + history.month / 12.0
        history = history.round({'date': 2})
        final_sim = int(res_df.iloc[len(res_df) - 1].Iteration)
        output_file = os.path.join(outerdir, site_name,
                               'CENTURY_outputs_iteration{}'.format(final_sim),
                               '{}.lis'.format(site_name))
        biomass_df = cent.read_CENTURY_outputs(output_file, 2015, 2015)
        biomass_df['time'] = biomass_df.index
        sum_df = history.merge(biomass_df, left_on='date', right_on='time',
                               how='inner')
        sum_df['site'] = site_name
        df_list.append(sum_df)
    summary_df = pandas.concat(df_list)
    summary_df.to_csv(save_as)

def combine_empirical_density_files(save_as):
    outerdir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\From_Sharon_5.29.15\Matched_GPS_records\Matched_with_spray_races'
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
        kamok = {'name': 'Kamok', 'biomass': 119.72, 'date': 2015.92}
        loidien = {'name': 'Loidien', 'biomass': 93.74, 'date': 2015.92}
        research = {'name': 'Research', 'biomass': 276.03, 'date': 2016.00}
        loirugurugu = {'name': 'Loirugu', 'biomass': 96.76, 'date': 2015.75}
        serat = {'name': 'Serat', 'biomass': 145.7, 'date': 2016.00}
        rongai = {'name': 'Rongai', 'biomass': 165.64, 'date': 2016.00}
    site_list = [research, loirugurugu]  # [kamok, loidien, research, loirugurugu, serat, rongai]
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
    return site_list
    
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
    emp_list = [r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\OPC_veg_9.30.16_by_weather.csv",
                r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\OPC_veg_9.30.16_by_weather_unrestricted_by_trees_shrubs.csv"]
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
            sim_dir = os.path.join(sim_outerdir, site['name'])
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
        save_as = os.path.join(save_dir, 'comparison_{}'.format(
                                              os.path.basename(empirical_csv)))
        df.to_csv(save_as)
    
if __name__ == "__main__":
    # run_empirical_stocking_density()
    save_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\OPC_integrated_test\empirical_stocking_density"
    summarize_comparison(save_dir)
    