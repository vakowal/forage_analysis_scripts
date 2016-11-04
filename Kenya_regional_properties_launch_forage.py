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

def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""
    
    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)

def default_forage_args():
    """Default args for the forage model for regional properties."""
    
    forage_args = {
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2014,
            'start_month': 1,
            'num_months': 24,
            'mgmt_threshold': 0.1,
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GL',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            }
    return forage_args

def id_failed_simulation(result_dir, num_months):
    """Test whether a simulation completed the specified num_months. Returns 0
    if the simulation failed, 1 if the simulation succeeded."""
    
    try:
        sum_csv = os.path.join(result_dir, 'summary_results.csv')
        sum_df = pd.read_csv(sum_csv)
    except:
        return 0
    if len(sum_df) == (num_months + 1):
        return 1
    else:
        return 0    

def run_preset_densities():
    """Run a series of stocking densities at each regional property."""
    
    failed = []
    template_herb_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herd_avg_uncalibrated.csv"
    density_list = [0.05, 0.475, 1., 2., 3., 5.]  # cattle per ha
    
    marg_dict = {'site': [], 'density': [], 'avg_yearly_gain': [],
                 'total_delta_weight_kg': []}
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\preset_dens_uncalibrated"
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015"
    forage_args = default_forage_args()
    forage_args['input_dir'] = input_dir
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['restart_monthly'] = 1
    for density in density_list:
        modify_stocking_density(template_herb_csv, density)
        for site in site_list:
            grass_csv = os.path.join(input_dir,
                                     '{:d}.csv'.format(int(site['name'])))
            add_cp_to_grass_csv(grass_csv)
            outdir = os.path.join(outer_outdir,
                                  'site_{:d}_{}_per_ha'.format(
                                  int(site['name']), density))
            forage_args['grass_csv'] = grass_csv
            forage_args['latitude'] = site['lat']
            forage_args['outdir'] = outdir
            if not os.path.exists(outdir):
                try:
                    forage.execute(forage_args)
                except:
                    continue
            succeeded = id_failed_simulation(outdir,
                                             forage_args['num_months'])
            if not succeeded:
                failed.append('site_{:d}_{}_per_ha'.format(int(site['name']),
                              density))
            else:
                sum_csv = os.path.join(outdir, 'summary_results.csv')
                sum_df = pd.read_csv(sum_csv)
                subset = sum_df.loc[sum_df['year'] > 2013]
                grouped = subset.groupby('year')
                avg_yearly_gain = (grouped.sum()['cattle_gain_kg']).mean()
                start_wt = sum_df.iloc[0]['cattle_kg']
                avg_yearly_gain_herd = avg_yearly_gain * float(density)
                perc_gain = avg_yearly_gain / float(start_wt)
                marg_dict['site'].append(site['name'])
                marg_dict['density'].append(density)
                marg_dict['avg_yearly_gain'].append(avg_yearly_gain)
                marg_dict['total_delta_weight_kg'].append(avg_yearly_gain_herd)
    df = pd.DataFrame(marg_dict)
    summary_csv = os.path.join(outer_outdir, 'gain_summary.csv')
    df.to_csv(summary_csv)
    erase_intermediate_files(outer_outdir)
    if len(failed) > 0:
        print "the following sites failed:"
        print failed

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
                    
def summarize_match(match_csv, save_as):
    """summarize the results of back-calc management routine by collecting
    the final comparison of empirical vs simulated biomass from each site
    where the routine was run."""
    
    sum_dict = {'year': [], 'site': [], 'live_or_total': [], 'g_m2': [],
                'sim_vs_emp': []}
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    for live_or_total in ['total', 'live']:
        for year_to_match in [2014, 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            for site in site_list:
                site_name = site['name']
                site_dir = os.path.join(outer_outdir,
                                        'back_calc_{}_{}'.format(year_to_match,
                                        live_or_total),
                                        'FID_{}'.format(site_name))
                result_csv = os.path.join(site_dir,
                                          'modify_management_summary_{}.csv'.
                                          format(site_name))
                res_df = pd.read_csv(result_csv)
                sum_dict['year'].extend([year_to_match] * 2)
                sum_dict['site'].extend([site_name] * 2)
                sum_dict['live_or_total'].extend([live_or_total] * 2)
                sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                        Simulated_biomass)
                sum_dict['sim_vs_emp'].append('sim')
                sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                        Empirical_biomass)
                sum_dict['sim_vs_emp'].append('emp')
    sum_df = pd.DataFrame(sum_dict)
    sum_df.to_csv(save_as)

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
    
    forage_args = default_forage_args()
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        outdir = os.path.join(outer_output_dir,
                              '{:d}'.format(int(site['name'])))
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        forage_args['latitude'] = site['lat']
        forage_args['outdir'] = outdir
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = None
        forage_args['input_dir'] = input_dir
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
    
    n_months = 24
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
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
            backcalc.summarize_calc_schedules(site_list, n_months, input_dir, 
                                              outerdir, raw_file, summary_file)
    
def add_cp_to_grass_csv(csv_file):
    """Modify the crude protein content in the grass csv used as input to the
    forage model."""
    
    df = pd.read_csv(csv_file)
    df['index'] = 0
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one grass type"
    df['cprotein_green'] = df['cprotein_green'].astype(float)
    df.set_value(0, 'cprotein_green', 0.14734)
    df['cprotein_dead'] = df['cprotein_dead'].astype(float)
    df.set_value(0, 'cprotein_dead', 0.014734)
    df.to_csv(csv_file)

if __name__ == "__main__":
    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    # run_baseline(site_csv)
    # combine_summary_files(site_csv)
    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Felicia\regional_veg_match_file.csv"
    # back_calc_mgmt(match_csv)
    # summarize_sch_wrapper(match_csv)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\back_calc_match_summary.csv"
    # summarize_match(match_csv, save_as)
    run_preset_densities()
