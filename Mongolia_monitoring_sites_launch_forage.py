# launch forage model for Boogie's monitoring sites in Mongolia

import os
import sys
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
            'num_months': 12,
            'mgmt_threshold': 0.01,
            'input_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_inputs',
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
    df.set_value(0, 'label', label)
    df.to_csv(grass_csv)
    
def run_zero_sd(site_csv):
    """Run the model without animals to compare simulated biomass to Boogie's
    biomass as a first step."""
    
    forage_args = default_forage_args()
    forage_args['input_dir'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\no_grazing"
    modify_stocking_density(forage_args['herbivore_csv'], 0)
    outer_outdir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/zero_sd'
    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    for site in site_list:
        forage_args['latitude'] = site['latitude']
        forage_args['outdir'] = os.path.join(outer_outdir,
                                             '{}'.format(site['site']))
        edit_grass_csv(forage_args['grass_csv'], site['site'])
        forage.execute(forage_args)
        
def run_avg_sd(site_csv):
    """Run the model with average animal density."""
    
    forage_args = default_forage_args()
    remove_months = [1, 2, 3, 11, 12]
    grz_months = range(0, forage_args['num_months'])
    for r in remove_months:
        grz_months = [m for m in grz_months if (m % 12) != (r - 1)]
    forage_args['grz_months'] = grz_months
    forage_args['input_dir'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\no_grazing"
    modify_stocking_density(forage_args['herbivore_csv'], 0.02)
    outer_outdir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/empirical_sd'
    site_list = pd.read_csv(site_csv).to_dict(orient='records')
    # for site in site_list:
    site = site_list[0]
    import pdb; pdb.set_trace()  # verify st17
    forage_args['latitude'] = site['latitude']
    forage_args['outdir'] = os.path.join(outer_outdir,
                                         '{}'.format(site['site']))
    edit_grass_csv(forage_args['grass_csv'], site['site'])
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
        
if __name__ == "__main__":
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\model_inputs\sites_median_grass_forb_biomass.csv"
    # run_zero_sd(site_csv)
    run_avg_sd(site_csv)
    save_as = r'C:/Users/Ginger/Dropbox/NatCap_backup/Mongolia/model_results/avg_sd/biomass_summary.csv'
    # compare_biomass(site_csv, save_as)
    