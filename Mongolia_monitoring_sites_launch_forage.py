# launch forage model for Boogie's monitoring sites in Mongolia

import os
import sys
import shutil
import pandas as pd
sys.path.append(
    "C:/Users/ginge/Documents/Python/rangeland_production_11_26_18")
# import forage
import back_calculate_management as backcalc

def default_forage_args():
    """Default args to run the forage model in Mongolia."""

    forage_args = {
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2016,
            'start_month': 1,
            'num_months': 24,
            'mgmt_threshold': 0.01,
            'century_dir': 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/cashmere_goats.csv",
            'grass_csv': r"C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/grass.csv",
            'restart_monthly': 0,
            }
    return forage_args

def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""

    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)

def edit_grass_csv(grass_csv, label):
    """Edit the grass csv to reflect a new label, which points to the Century
    inputs, so we can use one grass csv for multiple sites.  Century inputs
    must exist."""

    df = pd.read_csv(grass_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one grass type"
    df.set_value(0, 'label', label)
    df[['label']] = df[['label']].astype(type(label))
    df.to_csv(grass_csv)

def run_zero_sd(site_csv):
    """Run the model without animals to compare simulated biomass to Boogie's
    biomass as a first step."""
    # 'worldclim': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\Worldclim",
                              # r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\worldclim\zero_sd"],

    # run_dict = {'namem': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\soum_centers\namem_clim",
                          # r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\soum_centers\namem_clim\zero_sd"],
                # 'chirps': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\soum_centers\chirps_prec",
                           # r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\soum_centers\chirps_prec\zero_sd"]}
    # run_dict = {'chirps': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\CHIRPS_pixels\chirps_prec",
                           # r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\CHIRPS_pixels\chirps_prec\zero_sd"]}
    # run_dict = {'namem': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\soum_centers\namem_clim_wc_temp",
    #                       r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\soum_centers\namem_clim_wc_temp\zero_sd"]}
    run_dict = {'chirps': [r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\SCP_sites\chirps_prec",
                          r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites\chirps_prec\zero_sd_temp_calibrate"]}  # zero_sd

    forage_args = default_forage_args()
    modify_stocking_density(forage_args['herbivore_csv'], 0)
    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    for precip_source in run_dict.keys():
        forage_args['input_dir'] = run_dict[precip_source][0]
        outer_outdir = run_dict[precip_source][1]
        for site in site_list:
            forage_args['latitude'] = site['latitude']
            forage_args['outdir'] = os.path.join(outer_outdir,
                                                 '{}'.format(site['site_id']))
            if not os.path.isfile(os.path.join(forage_args['outdir'],
                                               'summary_results.csv')):
                edit_grass_csv(forage_args['grass_csv'], site['site_id'])
                forage.execute(forage_args)

def run_avg_sd(site_csv):
    """Run the model with average animal density."""

    forage_args = default_forage_args()
    remove_months = [1, 2, 3, 11, 12]
    grz_months = range(0, forage_args['num_months'])
    for r in remove_months:
        grz_months = [m for m in grz_months if (m % 12) != (r - 1)]
    forage_args['grz_months'] = grz_months
    modify_stocking_density(forage_args['herbivore_csv'], 0.02)
    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    run_dict = {'worldclim': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\Worldclim",
                              r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\worldclim\average_sd"],
                'namem': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\namem_clim",
                          r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\namem_clim\average_sd"]}
    for precip_source in run_dict.keys():
        forage_args['input_dir'] = run_dict[precip_source][0]
        outer_outdir = run_dict[precip_source][1]
        for site in site_list:
            forage_args['latitude'] = site['latitude']
            forage_args['outdir'] = os.path.join(outer_outdir,
                                                 '{}'.format(int(site['site_id'])))
            if not os.path.exists(forage_args['outdir']):
                edit_grass_csv(forage_args['grass_csv'], site['site_id'])
                forage.execute(forage_args)

def compare_biomass(site_csv, save_as):
    """Compare simulated and empirical biomass at Boogie's monitoring sites."""

    outer_outdir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/avg_sd'
    emp_list = pd.read_csv(site_csv)
    emp_list.set_index(['site'], inplace=True)
    emp_list['sim_gm2'] = ""
    for site in emp_list.index:
        outdir = os.path.join(outer_outdir, site)
        sim_df = pd.read_csv(os.path.join(outdir, 'summary_results.csv'))
        sim_df.set_index(['step'], inplace=True)
        # should measure July or August biomass according to empirical date,
        # but I am lazy: just use Sept 1
        sim_biom_kgha = sim_df.get_value(8, '{}_dead_kgha'.format(site)) + \
                        sim_df.get_value(8, '{}_green_kgha'.format(site))
        sim_biom_gm2 = sim_biom_kgha / 10
        emp_list.set_value(site, 'sim_gm2', sim_biom_gm2)
    emp_list.to_csv(save_as)

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

def clean_up():
    run_dict = {'chirps': [r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\SCP_sites\chirps_prec",
                          r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites\chirps_prec\zero_sd"]}
    for precip_source in run_dict.keys():
        outer_outdir = run_dict[precip_source][1]
        erase_intermediate_files(outer_outdir)

def summarize_biomass(site_csv, save_as):
    """Make a table of simulated biomass that can be compared to empirical
    biomass."""

    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    df_list = []
    outerdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites"
    columns = ['green_biomass_gm2', 'dead_biomass_gm2', 'total_biomass_gm2',
               'year', 'month']
    for precip_source in ['chirps_prec']:  # 'namem_clim', 'worldclim',
        for sd in ['zero_sd']:  # , 'average_sd']
            results_dir = os.path.join(outerdir, precip_source, sd)
            for site in site_list:
                site_id = site['site_id']
                sum_csv = os.path.join(results_dir, '{}'.format(site_id),
                                      'summary_results.csv')
                sum_df = pd.read_csv(sum_csv)
                sum_df = sum_df.set_index('step')
                subset = sum_df  # sum_df.loc[sum_df['month'].isin([7, 8, 9])]
                subset['green_biomass_gm2'] = subset['{}_green_kgha'.format(site_id)] / 10.
                subset['dead_biomass_gm2'] = subset['{}_dead_kgha'.format(site_id)] / 10.
                subset['total_biomass_gm2'] = subset['green_biomass_gm2'] + subset['dead_biomass_gm2']
                subset = subset[columns]
                subset['climate_source'] = precip_source
                subset['stocking_density_option'] = sd
                subset['site_id'] = site_id
                df_list.append(subset)
    sum_df = pd.concat(df_list)
    sum_df.to_csv(save_as)

def calc_variability(site_csv, save_as):
    """Collect minimum and maximum predicted forage for each site"""

    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    df_list = []
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\CHIRPS_pixels"
    sum_dict = {'site_id': [], 'min_biomass': [], 'max_biomass': []}
    for precip_source in ['chirps_prec']:  # 'namem_clim', 'worldclim',
        for sd in ['zero_sd']:  # , 'average_sd']
            results_dir = os.path.join(outerdir, precip_source, sd)
            for site in site_list:
                site_id = int(site['site_id'])
                sum_csv = os.path.join(results_dir, '{}'.format(site_id),
                                      'summary_results.csv')
                sum_df = pd.read_csv(sum_csv)
                sum_df = sum_df.set_index('step')
                biomass = sum_df['{}_green_kgha'.format(site_id)] + \
                                         sum_df['{}_dead_kgha'.format(site_id)]
                min_biom = min(biomass)
                max_biom = max(biomass)
                sum_dict['site_id'].append(site_id)
                sum_dict['min_biomass'].append(min_biom)
                sum_dict['max_biomass'].append(max_biom)
    sum_df = pd.DataFrame(sum_dict)
    sum_df.to_csv(save_as)


def back_calc(match_csv, input_dir, outdir):
    """Back calculate management to match 2016 empirical biomass."""
    century_dir = r'C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    fix_file = 'drygfix.100'
    n_months = 60
    vary = 'both'
    live_or_total = 'total'
    threshold = 5.0
    max_iterations = 1  # 50
    template_level = 'GGS'

    site_df = pd.read_csv(match_csv)
    for site in site_df.site_id.unique():
        sub_df = site_df.loc[site_df.site_id == site]
        assert len(sub_df) < 2, "must be only one site record to match"
        out_dir_site = os.path.join(outdir, site)
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site)
        site_dict = {
            'name': site, 'date': sub_df.date.values[0],
            'biomass': sub_df.biomass_g_m2.values[0]}
        if not os.path.exists(
                os.path.join(out_dir_site,
                'modify_management_summary_{}.csv'.format(site))):
            backcalc.back_calculate_management(
                site_dict, input_dir, century_dir, out_dir_site, fix_file,
                n_months, vary, live_or_total, threshold, max_iterations,
                template_level)


def collect_century_biomass_all_sites(match_csv, outdir, save_as):
    """Collect biomass outputs from Century for all sites in `match_csv`.

    Assume that relevant outputs are in the folder from the first iteration of
    the back-calc management routine, i.e. the starting schedule submitted to
    the routine. Collect live and standing dead biomass from 2016.

    """
    df_list = []
    site_df = pd.read_csv(match_csv)
    for site in site_df.site_id.unique():
        cent_file = os.path.join(
            outdir, site, 'CENTURY_outputs_iteration0', '{}.lis'.format(site))
        if not os.path.exists(cent_file):
            continue
        cent_df = pd.io.parsers.read_fwf(cent_file, skiprows=[1])
        df_subset = cent_df[(cent_df.time > 2016) & (cent_df.time <= 2017)]
        biomass_df = df_subset[['time', 'aglivc', 'stdedc']]
        live_biomass = biomass_df.aglivc * 2.5  # grams per square m
        dead_biomass = biomass_df.stdedc * 2.5  # grams per square m
        biomass_df['live'] = live_biomass
        biomass_df['standing_dead'] = dead_biomass
        biomass_df['total'] = live_biomass + dead_biomass
        biomass_df['site'] = site
        biomass_df.set_index('time', inplace=True)
        df_list.append(biomass_df)
    combined_df = pd.concat(df_list)
    combined_df.to_csv(save_as)


def summarize_back_calc_biomass(match_csv, outdir, save_as):
    """Summarize empirical/simulated biomass at starting/ending iterations."""
    df_list = []
    site_df = pd.read_csv(match_csv)
    for site in site_df.site_id.unique():
        out_csv = os.path.join(
            outdir, site, 'modify_management_summary_{}.csv'.format(site))
        if not os.path.exists(out_csv):
            continue
        try:
            out_df = pd.read_csv(out_csv)
        except pd.errors.EmptyDataError:  # back-calc did not complete
            continue
        max_iteration = max(out_df.Iteration)
        out_df_subset = out_df[
            (out_df.Iteration == 0) | (out_df.Iteration == max_iteration)]
        out_df_subset['site'] = site
        out_df_subset.set_index('site', inplace=True)
        df_list.append(out_df_subset)
    combined_df = pd.concat(df_list)
    combined_df.to_csv(save_as)


if __name__ == "__main__":
    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\sites_median_grass_forb_biomass.csv"
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\data\soil\soum_ctr_soil_isric_250m.csv"  # r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\data\soil\SCP_monitoring_points_soil_isric_250m.csv"
    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\data\soil\CHIRPS_pixels_soil_isric_250m.csv"
    site_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\data\soil\monitoring_points_soil_isric_250m.csv"
    # run_zero_sd(site_csv)
    # run_avg_sd(site_csv)
    # save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/avg_sd/biomass_summary.csv'
    # compare_biomass(site_csv, save_as)
    # save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/CHIRPS_pixels/biomass_summary_zero_sd_chirps_GCD_G.csv'
    # save_as = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites\chirps_prec\zero_sd\biomass_summary_zero_sd_chirps_GCD_G.csv"
    # summarize_biomass(site_csv, save_as)
    # clean_up()
    # save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/CHIRPS_pixels/min_max_biomass.csv'
    # calc_variability(site_csv, save_as)

    input_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\SCP_sites\chirps_prec_back_calc"
    outdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites\chirps_prec_back_calc"

    match_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/summaries_GK/herbaceous_biomass_2016_SCP_CBM.csv"
    match_csv = "C:/Users/ginge/desktop/missing_sites_backcalc.csv"  # hack
    back_calc(match_csv, input_dir, outdir)
    # comparison_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\monitoring_sites\chirps_prec_back_calc\match_summary.csv"
    # summarize_back_calc_biomass(match_csv, outdir, comparison_csv)
