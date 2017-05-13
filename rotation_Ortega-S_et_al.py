# rotation case study: Ortega-S et al 2013

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas as pd
import numpy as np
import rotation
sys.path.append(
 'C:/Users/Ginger/Documents/Python/rangeland_production')
import forage_century_link_utils as cent
import forage_utils as forage_u
import forage


def default_forage_args():
    """Default args for the forage model for Ortega-S et al test."""
    
    forage_args = {
            'latitude': 27.5,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2001,
            'start_month': 1,
            'num_months': 60,
            'mgmt_threshold': 0.1,
            'input_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al',
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'gfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al/cattle.csv",
            'grass_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Ortega-S_et_al/grass.csv",
            'digestibility_flag': 'Konza',
            'restart_monthly': 1,
            }
    return forage_args

def control(num_animals, total_area_ha, outdir):
    """Run the control scenario for the Ortega-S et al 2013 comparison:
    continuous year-long grazing at constant ranch-wide density."""
    
    winter_flag = 0
    forage_args = default_forage_args()
    rotation.continuous(forage_args, total_area_ha, num_animals, outdir,
                        winter_flag)

def blind_treatment(num_animals, total_area_ha, n_pastures, outdir):
    """Run blind rotation."""
    
    forage_args = default_forage_args()
    winter_flag = 0  # no grazing in months 12 or 1
    rotation.blind_rotation(forage_args, total_area_ha, n_pastures, num_animals,
                            outdir, winter_flag)
                      
def treatment(num_animals, n_pastures, pasture_size_ha, outdir):
    """Run "smart" rotation."""
    
    forage_args = default_forage_args()
    rotation.rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
                      outdir)

def stocking_dens_test_wrapper(outer_outdir):
    """Calculate difference in pasture and animal gain metrics between rotation
    and continuous grazing, at a range of total stocking densities."""
    
    # from Ortega-S et al 2013
    # n_pastures = 8
    total_area_ha = 48.56
    # num_animals = 14
    
    result_dict = {'num_animals': [], 'num_pastures': [], 'gain_%_diff': [],
                   'pasture_%_diff': []}
    for num_animals in [14, 21]:
        for n_pastures in[6, 8, 10, 12, 14]:
            cont_dir = os.path.join(outer_outdir,
                                    'cont_{}_animals'.format(num_animals))
            if not os.path.exists(os.path.join(cont_dir,
                                               'summary_results.csv')):
                control(num_animals, total_area_ha, cont_dir)
            rot_dir = os.path.join(outer_outdir,
                                   'blind_rot_{}_animals_{}_pastures'.format(
                                   num_animals, n_pastures))
            if not os.path.exists(os.path.join(rot_dir,
                                               'pasture_summary.csv')):
                blind_treatment(num_animals, total_area_ha, n_pastures,
                                rot_dir)
            gain_diff, pasture_diff = rotation.calc_productivity_metrics(
                                                             cont_dir, rot_dir)
            result_dict['num_animals'].append(num_animals)
            result_dict['num_pastures'].append(n_pastures)
            result_dict['gain_%_diff'].append(gain_diff)
            result_dict['pasture_%_diff'].append(pasture_diff)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(outer_outdir, 'summary.csv'))

def get_biomass(outdir, date):
    """Retrieve biomass from model outputs on the specified date. Date must be
    in "Century format", e.g. 2012.00 is December of 2011."""
    
    # find latest folder of outputs
    folder_list = [f for f in os.listdir(outdir) if
                    os.path.isdir(os.path.join(outdir, f))]
    folder_list.remove('CENTURY_outputs_spin_up')
    max_year = max([f[-4:] for f in folder_list])
    folder_list = [f for f in folder_list if f.endswith(max_year)]
    max_month = max([int(re.search(
                    'CENTURY_outputs_m(.+?)_y{}'.format(max_year),
                    f).group(1)) for f in folder_list])
    max_folder = os.path.join(outdir,
                              'CENTURY_outputs_m{}_y{}'.format(
                              max_month, max_year))
    output_files = [f for f in os.listdir(max_folder) if f.endswith('.lis')]
    output_f = os.path.join(max_folder, output_files[0])
    outputs = cent.read_CENTURY_outputs(output_f, math.floor(date) - 1,
                                        math.ceil(date) + 1)
    outputs.drop_duplicates(inplace=True)
    live_gm2 = outputs.loc[date, 'aglivc']
    dead_gm2 = outputs.loc[date, 'stdedc']
    return live_gm2, dead_gm2

def get_date(forage_args, step):
    """Calculate the "Century date" corresponding to a model step. Model step
    is relative to starting month of the model, e.g. step 0 is first step of
    the model run, or forage_args['start_year'], forage_args['start_month']"""
    
    step_month = forage_args[u'start_month'] + step
    if step_month > 12:
        mod = step_month % 12
        if mod == 0:
            month = 12
            year = (step_month / 12) + forage_args[u'start_year'] - 1
        else:
            month = mod
            year = (step_month / 12) + forage_args[u'start_year']
    else:
        month = step_month
        year = (step / 12) + forage_args[u'start_year']
    date = year + float('%.2f' % (month / 12.))
    return date

def diet_selection(forage_args, live_gm2, dead_gm2, defoliation_step):
    """Perform diet selection on forage, as a metric of forage quality. Return
    diet selected by an individual animal from the forage available on the date
    defoliation_step.  defoliation_step should be supplied relative to model
    steps, e.g. step 0 is first step of the model run, or
    forage_args['start_year'], forage_args['start_month']"""
    
    forage_u.set_time_step('month')
    herbivore_list = []
    herbivore_input = (pd.read_csv(forage_args['herbivore_csv']).to_dict(
                       orient='records'))
    for h_class in herbivore_input:
        herd = forage_u.HerbivoreClass(h_class)
        herd.update()
        herbivore_list.append(herd)
    stocking_density_dict = forage_u.populate_sd_dict(herbivore_list)
    total_SD = forage_u.calc_total_stocking_density(herbivore_list)
    grass_list = (pd.read_csv(forage_args['grass_csv'])).to_dict(orient='records')
    assert len(grass_list) == 1, "Must be one grass type"
    grass_list[0]['green_gm2'] = live_gm2
    grass_list[0]['dead_gm2'] = dead_gm2
    available_forage = forage_u.calc_feed_types(grass_list)
    for feed_type in available_forage:
        feed_type.calc_digestibility_from_protein()  # TODO n_mult??
    assert len(herbivore_list) == 1, "must be one herbivore type"
    herb_class = herbivore_list[0]
    herb_class.calc_distance_walked(1., total_SD, available_forage)
    max_intake = herb_class.calc_max_intake()
    ZF = herb_class.calc_ZF()
    HR = forage_u.calc_relative_height(available_forage)
    diet = forage_u.diet_selection_t2(ZF, HR, 0., 0, max_intake,
                                    herb_class.FParam,
                                    available_forage,
                                    herb_class.f_w, herb_class.q_w)
    diet_interm = forage_u.calc_diet_intermediates(diet, herb_class,
                                                 0., 150)
    reduced_max_intake = forage_u.check_max_intake(diet, diet_interm,
                                                 herb_class,
                                                 max_intake)
    if reduced_max_intake < max_intake:
        diet = forage_u.diet_selection_t2(ZF, HR, 0., 0, max_intake,
                                        herb_class.FParam,
                                        available_forage,
                                        herb_class.f_w,
                                        herb_class.q_w)
    return diet
                                                
def optimal_defoliation_test():
    """Test grass response to different grazing levels explicitly.  Confirm
    that optimal defolation level for biomass regrowth is 25%"""
    
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\WitW\Ortega-S_et_al\defol_exp"
    forage_args = default_forage_args()
    
    result_dict = {'defoliation_step': [], 'defoliation_level': [],
                   'live_biomass_prior_to_defol': [],
                   'live_biomass_next_step': []}
    for defol_step in [6]:  # start with one month
        # target_month is specified relative to steps, so grz_months
        forage_args['grz_months'] = []
        forage_args['outdir'] = os.path.join(outer_outdir, 'zero_sd')
        if not os.path.exists(os.path.join(forage_args['outdir'],
                                           'summary_results.csv')):
            forage.execute(forage_args)
        defol_date = get_date(forage_args, defol_step)
        live_gm2, dead_gm2 = get_biomass(forage_args['outdir'], defol_date)
        
        # calculate diet selected in defoliation month: then can change stocking density to 
        # remove specified amount of biomass
        diet = diet_selection(forage_args, live_gm2, dead_gm2, defol_step)  # daily intake
        monthly_intake_live = forage_u.convert_daily_to_step(diet.intake['grass;green'])
        monthly_intake_live_gm2 = monthly_intake_live / 10.
        for defol_level in [0.1, 0.15]:  # , 0.2, 0.25, 0.3, 0.35]:
            # for now, calculate stocking dens based on % live biomass consumed
            live_gm2_consumed = defol_level * live_gm2
            stocking_dens = live_gm2_consumed / monthly_intake_live_gm2
            rotation.modify_stocking_density(forage_args['herbivore_csv'],
                                             stocking_dens)
            outdir = os.path.join(outer_outdir,
                                  'defol_{0:.2f}'.format(defol_level))
            forage_args['outdir'] = outdir
            forage_args['grz_months'] = [defol_step]
            forage.execute(forage_args)
            regrowth_step = defol_step + 1
            regrowth_date = get_date(forage_args, regrowth_step)
            # record biomass in following step
            live_gm2_after, dead_gm2_after = get_biomass(outdir, defol_date)
            result_dict['defoliation_step'].append(defol_step)
            result_dict['defoliation_level'].append(defol_level)
            result_dict['live_biomass_prior_to_defol'].append(live_gm2)
            result_dict['live_biomass_next_step'].append(live_gm2_after)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(outer_outdir, 'regrowth_summary.csv'))
    
if __name__ == "__main__":
    optimal_defoliation_test()
