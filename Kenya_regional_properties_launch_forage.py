# run forage model on regional properties

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas as pd
import numpy as np
import back_calculate_management as backcalc
sys.path.append(
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_century_link_utils as cent
import forage

def back_calc_mgmt(match_csv):
    """Use the back-calc management routine to calculate management at regional
    properties prior to the 2014 or 2015 measurement."""
    
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    for live_or_total in ['total', 'live']:
        for year_to_match in [2014, 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
            n_months = 24
            century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
            out_dir = os.path.join(outer_outdir, "back_calc_{}_{}".format(
                                                 year_to_match, live_or_total))
            vary = 'both'
            threshold = 10.0
            max_iterations = 40
            fix_file = 'drytrpfi.100'
            for site in site_list:
                out_dir_site = os.path.join(out_dir, 'FID_{}'.format(
                                                                 site['name']))
                if not os.path.exists(out_dir_site):
                    os.makedirs(out_dir_site) 
                    backcalc.back_calculate_management(site, input_dir,
                                                       century_dir, out_dir_site,
                                                       fix_file, n_months,
                                                       vary, live_or_total,
                                                       threshold, max_iterations)

def generate_inputs(match_csv, year_to_match, live_or_total):
    """Generate a list that can be used as input to run the back-calc
    management routine.  Year_to_match should be 2014 or 2015.  live_or_total
    should be 'live' or 'total'."""
    
    site_list = []
    site_df = pd.read_csv(match_csv)
    for site in site_df.Property.unique():
        sub_df = site_df.loc[(site_df.Property == site) & (site_df.Year == year_to_match)]
        assert len(sub_df) < 2, "must be only one site record to match"
        if len(sub_df) == 0:
            continue
        site_name = str(sub_df.get_value(sub_df.index[0], 'FID'))
        year, month = str((sub_df.sim_date).values[0]).rsplit("-")[:2]
        century_date = round((int(year) + int(month) / 12.0), 2)
        total_biomass_emp = sub_df.get_value(sub_df.index[0], 'TotalBiomass')
        green_biomass_emp = sub_df.get_value(sub_df.index[0], 'GBiomass')
        site_dict = {'name': site_name, 'date': century_date}
        if live_or_total == 'live':
            site_dict['biomass'] = green_biomass_emp
        elif live_or_total == 'total':
            site_dict['biomass'] = total_biomass_emp
        site_list.append(site_dict)
    return site_list
        
def run_baseline(site_csv):
    """Run the model with zero grazing, for each regional property."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        outdir = os.path.join(outer_output_dir,
                              '{:d}'.format(int(site['name'])))
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        forage_args = {
            'latitude': site['lat'],
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2014,
            'start_month': 1,
            'num_months': 24,
            'mgmt_threshold': 0.1,
            'century_dir': century_dir,
            'outdir': outdir,
            'template_level': 'GL',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 0,
            'user_define_digestibility': 0,
            'herbivore_csv': None,
            'grass_csv': grass_csv,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            'input_dir': input_dir,
            }
        forage.execute(forage_args)

def combine_summary_files(site_csv):
    """Make a file that can be used to plot biomass differences between
    sites."""
    
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip\combined_summary.csv"
    df_list = []
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        sim_name = int(site['name'])
        sim_dir = os.path.join(outer_output_dir, '{}'.format(sim_name))
        sim_df = pd.read_csv(os.path.join(sim_dir,'summary_results.csv'))
        sim_df['total_kgha'] = sim_df['{}_green_kgha'.format(sim_name)] + \
                                    sim_df['{}_dead_kgha'.format(sim_name)]
        sim_df['site'] = sim_name
        sim_df = sim_df[['site', 'total_kgha', 'month', 'year',
                         'total_offtake']]
        df_list.append(sim_df)
    combined_df = pd.concat(df_list)
    combined_df.to_csv(save_as)

def summarize_sch_wrapper(match_csv):
    """Wrapper function to summarize back-calculated schedules in several
    directories specified by year_to_match and live_or_total."""
    
    n_months = 12
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    for live_or_total in ['total', 'live']:
        for year_to_match in [2014, 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            outerdir = os.path.join(outer_outdir, "back_calc_{}_{}".format(
                                                 year_to_match, live_or_total))                            
            raw_file = os.path.join(outerdir,
                                   "{}_{}_schedule_summary.csv".format(
                                   year_to_match, live_or_total))
            summary_file = os.path.join(outerdir,
                                   "{}_{}_percent_removed.csv".format(
                                   year_to_match, live_or_total))
            summarize_calc_schedules(site_list, n_months, outerdir, raw_file,
                                     summary_file)
    
    
def summarize_calc_schedules(site_list, n_months, outerdir, raw_file,
                             summary_file):
    """summarize the grazing schedules that were calculated via the back-calc
    management regime.
    We summarize grazing scheduled only during the two years that were modified
    as part of the back-calc regime.  We assume that relative year 1 in the
    schedule file is equal to 2011."""
    
    no_mod_list = []  # sites for which no modification was necessary
    df_list = list()
    for site in site_list:
        flgrem_GLP = 0.1
        fdgrem_GLP = 0.05
        flgrem_GL = 0.1
        fdgrem_GL = 0.05
        site_name = site['name']
        empirical_date = site['date']
        site_dir = os.path.join(outerdir, 'FID_{}'.format(site_name))
        result_csv = os.path.join(site_dir,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pd.read_csv(result_csv)
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [int(re.search('{}_{}(.+?).sch'.format(site_name,
                         site_name), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:
            no_mod_list.append(result_csv)
            continue  # TODO find original schedule
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
        first_rel_month, first_rel_year = cent.find_first_month_and_year(
                                               n_months, empirical_month,
                                               relative_empirical_year)
        first_abs_year = first_rel_year + start_year - 1
        # find months where grazing took place prior to empirical date
        graz_schedule = cent.read_graz_level(final_sch)
        block = graz_schedule.loc[(graz_schedule["block_end_year"] == 
                                  last_year), ['relative_year', 'month',
                                  'grazing_level', 'year']]
        empirical_year = block.loc[(block['relative_year'] ==
                                   relative_empirical_year) & 
                                   (block['month'] <= empirical_month), ]
        intervening_years = block.loc[(block['relative_year'] <
                                      relative_empirical_year) & 
                                      (block['relative_year'] > first_rel_year), ]
        first_year = block.loc[(block['relative_year'] == first_rel_year) & 
                                      (block['month'] >= first_rel_month), ]
        history =  pd.concat([first_year, intervening_years, empirical_year])
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
                                flgrem_GL = float(line[:8].strip())
                                line = old_file.next()
                            if 'FDGREM' in line:
                                fdgrem_GL = float(line[:8].strip())
                            else:
                                er = "Error: FLGREM expected"
                                raise Exception(er)
        
        # fill in history with months where no grazing took place
        history = cent.fill_schedule(history, first_rel_year, first_rel_month,
                                     relative_empirical_year, empirical_month)
        history['year'] = history['relative_year'] + start_year - 1
        history['perc_live_removed'] = np.where(
                                           history['grazing_level'] ==  'GL',
                                           flgrem_GL, np.where(
                                           history['grazing_level'] == 'GLP',
                                           flgrem_GLP, 0.))
        history['perc_dead_removed'] = np.where(
                                           history['grazing_level'] == 'GL',
                                           fdgrem_GL, np.where(
                                           history['grazing_level'] == 'GLP',
                                           fdgrem_GLP, 0.))
        # collect biomass for these months
        history['date'] = history.year + history.month / 12.0
        history = history.round({'date': 2})
        final_sim = int(res_df.iloc[len(res_df) - 1].Iteration)
        output_file = os.path.join(site_dir,
                               'CENTURY_outputs_iteration{}'.format(final_sim),
                               '{}.lis'.format(site_name))
        biomass_df = cent.read_CENTURY_outputs(output_file, first_abs_year,
                                               math.floor(empirical_date))
        biomass_df['time'] = biomass_df.index
        sum_df = history.merge(biomass_df, left_on='date', right_on='time',
                               how='inner')
        sum_df['site'] = site_name
        df_list.append(sum_df)
    summary_df = pd.concat(df_list)
    summary_df['live_rem'] = summary_df.aglivc * summary_df.perc_live_removed
    summary_df['dead_rem'] = summary_df.stdedc * summary_df.perc_dead_removed
    summary_df['total_rem'] = summary_df.live_rem + summary_df.dead_rem
    rem_means = summary_df.groupby(by='site')[('total_rem', 'live_rem',
                                               'dead_rem')].mean()
    rem_means.to_csv(summary_file)
    summary_df.to_csv(raw_file)

if __name__ == "__main__":
    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    # run_baseline(site_csv)
    # combine_summary_files(site_csv)
    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Felicia\regional_veg_match_file.csv"
    back_calc_mgmt(match_csv)
    summarize_sch_wrapper(match_csv)