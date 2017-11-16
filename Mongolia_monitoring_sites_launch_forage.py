# launch forage model for Boogie's monitoring sites in Mongolia

import os
import sys
import shutil
import pandas as pd
sys.path.append(
    "C:/Users/Ginger/Documents/Python/rangeland_production")
import forage


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
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_inputs/cashmere_goats.csv",
            'grass_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_inputs/grass.csv",
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
    df[['label']] = df[['label']].astype(type(label))
    df.set_value(0, 'label', label)
    df.to_csv(grass_csv)
    
def run_zero_sd(site_csv):
    """Run the model without animals to compare simulated biomass to Boogie's
    biomass as a first step."""
    
    run_dict = {'worldclim': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\Worldclim",
                              r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\worldclim\zero_sd"],
                'namem': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\namem_clim",
                          r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\namem_clim\zero_sd"]}
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
            if not os.path.exists(forage_args['outdir']):
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
                                                 '{}'.format(site['site_id']))
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
    run_dict = {'worldclim': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\Worldclim",
                              r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\worldclim\zero_sd"],
                'namem': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\namem_clim",
                          r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\namem_clim\zero_sd"], 
                'worldclim2': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\Worldclim",
                              r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\worldclim\average_sd"],
                'namem2': [r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\namem_clim",
                          r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results\namem_clim\average_sd"]}
    
    for precip_source in run_dict.keys():
        outer_outdir = run_dict[precip_source][1]
        erase_intermediate_files(outer_outdir)

def summarize_biomass(save_as):
    """Make a table of simulated biomass that can be compared to empirical
    biomass."""
    
    df_list = []
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_results"
    columns = ['green_biomass_gm2', 'dead_biomass_gm2', 'total_biomass_gm2',
               'year', 'month']
    for precip_source in ['worldclim', 'namem_clim']:
        for sd in ['zero_sd']:  # , 'average_sd']
            results_dir = os.path.join(outerdir, precip_source, sd)
            for folder in os.listdir(results_dir):
                site = os.path.basename(folder)
                sum_df = pd.read_csv(os.path.join(results_dir, folder,
                                                  'summary_results.csv'))
                sum_df = sum_df.set_index('step')
                subset = sum_df.loc[sum_df['month'].isin([7, 8, 9])]
                subset['green_biomass_gm2'] = subset['{}_green_kgha'.format(site)] / 10.
                subset['dead_biomass_gm2'] = subset['{}_dead_kgha'.format(site)] / 10.
                subset['total_biomass_gm2'] = subset['green_biomass_gm2'] + subset['dead_biomass_gm2']
                subset = subset[columns]
                subset['climate_source'] = precip_source
                subset['stocking_density_option'] = sd
                subset['site_id'] = site
                df_list.append(subset)
    sum_df = pd.concat(df_list)
    sum_df.to_csv(save_as)

if __name__ == "__main__":
    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\sites_median_grass_forb_biomass.csv"
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\data\soil\monitoring_points_soil_isric_250m.csv"
    # run_zero_sd(site_csv)
    # run_avg_sd(site_csv)
    # save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/avg_sd/biomass_summary.csv'
    # compare_biomass(site_csv, save_as)
    # clean_up()
    save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/biomass_summary_zero_sd_worldclim_namem.csv'
    summarize_biomass(save_as)
    