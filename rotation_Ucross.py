# test "composition" mechanism of rotation
# with case study at Ucross, Wyoming

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
    """Default args for the forage model for Ucross case study."""
    
    forage_args = {
            'latitude': 44.6,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2002,
            'start_month': 1,
            'num_months': 60,
            'mgmt_threshold': 0.01,
            'input_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross',
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross/cattle.csv",
            'grass_csv': r"C:/Users/Ginger/Dropbox/NatCap_backup/WitW/model_inputs/Ucross/grass.csv",
            'digestibility_flag': 'Konza',
            'restart_monthly': 1,
            }
    return forage_args

def edit_grass_csv(csv, high_cp, low_cp, high_perc, high_cp_label,
                   low_cp_label):
    """Edit grass descriptors in grass csv to reflect the given inputs in terms
    of crude protein content and relative biomass."""
    
    grass_df = pd.read_csv(csv)
    grass_df.set_index("label", inplace=True)
    grass_df.cprotein_green = grass_df.cprotein_green.astype(float)
    grass_df.cprotein_dead = grass_df.cprotein_dead.astype(float)
    grass_df.percent_biomass = grass_df.percent_biomass.astype(float)
    
    grass_df = grass_df.set_value(high_cp_label, 'cprotein_green', high_cp)
    grass_df = grass_df.set_value(high_cp_label, 'cprotein_dead', 0.7*high_cp)
    grass_df = grass_df.set_value(low_cp_label, 'cprotein_green', low_cp)
    grass_df = grass_df.set_value(low_cp_label, 'cprotein_dead', 0.7*low_cp)
    grass_df = grass_df.set_value(high_cp_label, 'percent_biomass', high_perc)
    grass_df = grass_df.set_value(low_cp_label, 'percent_biomass', (
                                  1 - high_perc))
    grass_df.to_csv(csv)

def composition_wrapper():
    """Beginning of a wrapper function to test effect of different inputs on
    benefit of rotation as mediated by changes in pasture composition."""
    
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross"
    comp_filename = os.path.join(outer_outdir, 'proportion_summary_10_130_250_anim.csv')
    # bene_filename = os.path.join(outer_outdir, 'benefit_rot_36_mo.csv')
    high_cp_label = 'high_quality'
    low_cp_label = 'low_quality'
    
    # fixed
    # num_animals = 10 # original from Ucross description: 350
    total_area_ha = 688
    cp_mean = 0.1545
    cp_ratio_list = [1.2]  # [1, 1.2] # , 1.2]
    high_quality_perc = 0.5
    
    # vary
    n_pasture_list = [3]  # , 4]
    # high_quality_perc_list = [0.5]  # , 0.2]  # , 0.4, 0.5, 0.6]
    num_animals_list = [10, 130, 250]
    
    df_list = []
    result_dict = {'n_pastures': [], 'num_animals': [],
                   'cp_ratio': [], 'gain_%_diff': []}
    remove_months = [1, 2, 3, 11, 12]
    for num_animals in num_animals_list:  # high_quality_perc in high_quality_perc_list:
        for cp_ratio in cp_ratio_list:
            low_quality_cp = (2. * cp_mean) / (cp_ratio + 1.)
            high_quality_cp = float(cp_ratio) * low_quality_cp
            forage_args = default_forage_args()
            edit_grass_csv(forage_args['grass_csv'], high_quality_cp,
                           low_quality_cp, high_quality_perc, high_cp_label,
                           low_cp_label)
            cont_dir = os.path.join(outer_outdir,
                                    'control_{}_anim_perc_{}'.format(
                                    num_animals, cp_ratio))
            if not os.path.exists(os.path.join(cont_dir,
                                  'summary_results.csv')):
                control(num_animals, total_area_ha, cont_dir, remove_months)
            for n_pastures in n_pasture_list:
                rot_dir = os.path.join(outer_outdir,
                                      'rot_{}_pastures_{}_anim_perc_{}'.format(
                                         n_pastures, num_animals, cp_ratio))
                if not os.path.exists(os.path.join(rot_dir,
                                      'pasture_summary.csv')):
                    blind_treatment(num_animals, total_area_ha, n_pastures,
                                    rot_dir, remove_months)
                diff_df = proportion_high_quality(cont_dir, rot_dir)
                diff_df['n_pastures'] = [n_pastures] * len(diff_df.index)
                diff_df['num_animals'] = [num_animals] * len(diff_df.index)
                diff_df['cp_ratio'] = [cp_ratio] * len(diff_df.index)
                df_list.append(diff_df)
                # gain_diff, pasture_diff = rotation.calc_productivity_metrics(
                                                                 # cont_dir, rot_dir)
                result_dict['n_pastures'].append(n_pastures)
                result_dict['num_animals'].append(num_animals)
                result_dict['cp_ratio'].append(cp_ratio)
                # result_dict['gain_%_diff'].append(gain_diff)
    composition_effect_df = pd.concat(df_list)
    composition_effect_df.to_csv(comp_filename)
    
    # result_df = pd.DataFrame(result_dict)
    # result_df.to_csv(bene_filename)

def proportion_high_quality(cont_dir, rot_dir):
    """contrast the proportion of high quality grass between continuous and
    rotated schedules. Grass must be labeled 'high_quality'"""
    
    cont_sum_df = pd.read_csv(os.path.join(cont_dir, 'summary_results.csv'))
    grass_type_cols = [c for c in cont_sum_df.columns.tolist() if
                       re.search('kgha', c)]
    total_grass = cont_sum_df[grass_type_cols].sum(axis=1, skipna=False)
    prop_highq_c = ((cont_sum_df['high_quality_green_kgha'] + 
             cont_sum_df['high_quality_dead_kgha']) /
                                                     total_grass).tolist()
                                          
    rot_sum_df = pd.read_csv(os.path.join(rot_dir, 'pasture_summary.csv'))
    rot_grp = rot_sum_df.groupby('step').mean()
    grass_cols = [c for c in rot_grp.columns.tolist() if
                       re.search('total_kgha', c)]
    total_grass = rot_grp[grass_cols].sum(axis=1, skipna=False)
    prop_highq_r = rot_grp['high_quality_total_kgha'] / total_grass.tolist()

    assert len(prop_highq_c) == len(prop_highq_r), "Continuous and rotation must have same length"
    prop_dict = dict()
    prop_dict['continuous'] = prop_highq_c
    prop_dict['rotation'] = prop_highq_r
    prop_dict['month'] = rot_grp['month'].tolist()
    prop_dict['year'] = rot_grp['year'].tolist()
    prop_dict['step'] = rot_grp.index.tolist()
    prop_df = pd.DataFrame(prop_dict)
    prop_df.set_index(['step'], inplace=True)
    return prop_df
    
def composition_diff(cont_dir, rot_dir):
    """What is difference between continuous results and rotation results in
    terms of pasture composition?  Metric of pasture composition is the
    difference between the proportions of two grass types."""
    
    cont_sum_df = pd.read_csv(os.path.join(cont_dir, 'summary_results.csv'))
    grass_type_cols = [c for c in cont_sum_df.columns.tolist() if
                       re.search('kgha', c)]
    total_grass = cont_sum_df[grass_type_cols].sum(axis=1, skipna=False)
    green_grass_cols = [c for c in grass_type_cols if
                        re.search('green_kgha', c)]
    grass_labels = [re.sub("_green_kgha", "", c) for c in green_grass_cols]
    assert len(grass_labels) == 2, "Can only handle 2 grass types"
    prop_dict = {}
    for label in grass_labels:
        prop = ((cont_sum_df['{}_green_kgha'.format(label)] + 
                 cont_sum_df['{}_dead_kgha'.format(label)]) /
                                                         total_grass).tolist()
        prop_dict[label] = prop
    prop_df = pd.DataFrame(prop_dict)
    
    # subtract from each other, in fixed order
    prop_diff_c = (prop_df[grass_labels[0]] -
                   prop_df[grass_labels[1]][0]).tolist()
                                          
    rot_sum_df = pd.read_csv(os.path.join(rot_dir, 'pasture_summary.csv'))
    rot_grp = rot_sum_df.groupby('step').mean()
    grass_cols = [c for c in rot_grp.columns.tolist() if
                       re.search('total_kgha', c)]
    total_grass = rot_grp[grass_cols].sum(axis=1, skipna=False)

    prop_dict = {}
    for g_col in grass_cols:
        prop = rot_grp[g_col] / total_grass.tolist()
        prop_dict[g_col] = prop
    prop_df = pd.DataFrame(prop_dict)
    
    # subtract from each other, in fixed order
    prop_diff_r = (prop_df[grass_cols[0]] - prop_df[grass_cols[1]][0]).tolist()

    assert len(prop_diff_c) == len(prop_diff_r), "Continuous and rotation must have same length"
    diff_dict = dict()
    diff_dict['continuous'] = prop_diff_c
    diff_dict['rotation'] = prop_diff_r
    diff_dict['month'] = rot_grp['month'].tolist()
    diff_dict['year'] = rot_grp['year'].tolist()
    diff_dict['step'] = rot_grp.index.tolist()
    diff_df = pd.DataFrame(diff_dict)
    diff_df.set_index(['step'], inplace=True)
    return diff_df
    
def control(num_animals, total_area_ha, outdir, remove_months=None):
    """Run the control scenario for the Ucross comparison:
    continuous year-long grazing at constant ranch-wide density."""
    
    forage_args = default_forage_args()
    rotation.continuous(forage_args, total_area_ha, num_animals, outdir,
                        remove_months)

def blind_treatment(num_animals, total_area_ha, n_pastures, outdir,
                    remove_months=None):
    """Run blind rotation."""
    
    forage_args = default_forage_args()
    rotation.blind_rotation(forage_args, total_area_ha, n_pastures, num_animals,
                            outdir, remove_months)

def stocking_dens_test_wrapper(outer_outdir):
    """Calculate difference in pasture and animal gain metrics between rotation
    and continuous grazing, at a range of total stocking densities."""
    
    # from Todd Graham's report
    total_area_ha = 688
    # num_animals = 350
    remove_months = [1, 2, 3, 11, 12]
    
    result_dict = {'num_animals': [], 'num_pastures': [], 'gain_%_diff': [],
                   'pasture_%_diff': []}
    for num_animals in [200, 350, 500]:
        for n_pastures in[2, 4, 6, 8, 10]:
            cont_dir = os.path.join(outer_outdir,
                                    'cont_{}_animals'.format(num_animals))
            if not os.path.exists(os.path.join(cont_dir,
                                               'summary_results.csv')):
                try:
                    control(num_animals, total_area_ha, cont_dir,
                            remove_months)
                except:
                    continue
            rot_dir = os.path.join(outer_outdir,
                                   'blind_rot_{}_animals_{}_pastures'.format(
                                   num_animals, n_pastures))
            if not os.path.exists(os.path.join(rot_dir,
                                               'pasture_summary.csv')):
                try:
                    blind_treatment(num_animals, total_area_ha, n_pastures,
                                    rot_dir, remove_months)
                except:
                    continue
            gain_diff, pasture_diff = rotation.calc_productivity_metrics(
                                                             cont_dir, rot_dir)
            result_dict['num_animals'].append(num_animals)
            result_dict['num_pastures'].append(n_pastures)
            result_dict['gain_%_diff'].append(gain_diff)
            result_dict['pasture_%_diff'].append(pasture_diff)
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(outer_outdir,
                                  'stocking_density_test_summary.csv'))

def zero_sd():
    """Run zero stocking density to get forage production right."""
    
    outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\zero_sd_KNZ_2"
    num_animals = 0
    total_area_ha = 1
    
    forage_args = default_forage_args()
    control(num_animals, total_area_ha, outdir)

def generate_grz_months_rest_period(total_steps, rest_period):
    """Generate grazing months to be supplied to the model. Start month is the
    first month of grazing (where month 0 is the first month of the
    simulation). Rest period is the number of months after each grazing month
    where grazing does not occur."""
    
    grz_months = [(rest_period + 1) * m for m in xrange(0, total_steps)]
    grz_months = [m for m in grz_months if m < total_steps]
    return grz_months

def rest_effect_wrapper():
    """Can we show that there are any beneficial effects to grazing, or to
    grazing with rest?"""
    
    # first run with no grazing
    outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\rest_effect\zero_sd"
    num_animals = 0
    total_area_ha = 1
    
    forage_args = default_forage_args()
    forage_args['grass_csv'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_inputs\Ucross\grass_high_quality_only.csv"
    rotation.continuous(forage_args, total_area_ha, num_animals, outdir)
    
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\rest_effect"
    remove_months = [1, 2, 3, 11, 12]
    sd_list = [0.25, 0.5, 0.75, 1]
    for sd in sd_list:
        num_animals = sd
        outdir = os.path.join(outer_dir, '{}_cont'.format(sd))
        rotation.continuous(forage_args, total_area_ha, num_animals, outdir,
                            remove_months)
        for rest_period in [1, 2, 3, 4]:
            forage_args['grz_months'] = generate_grz_months_rest_period(
                                            forage_args['num_months'],
                                            rest_period)
            outdir = os.path.join(outer_dir,
                                  '{}_rest_{}_per_ha'.format(rest_period, sd))
            forage_args['outdir'] = outdir
            rotation.modify_stocking_density(forage_args['herbivore_csv'], sd)
            forage.execute(forage_args)

def collect_results():
    """Very hack-y way to collect results from rest rotation experiment."""
    
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\rest_effect\summary_figs\rest_effect_summary.csv"
    
    def get_from_century_results(outdir, sd, treatment, rest_period):
        cent_file = os.path.join(outdir, 'CENTURY_outputs_m12_y2006', 'high_quality.lis')
        cent_df = pd.io.parsers.read_fwf(cent_file, skiprows = [1])
        df_subset = cent_df[(cent_df.time >= 2002) & (cent_df.time <= 2007)]
        outputs = df_subset[['time', 'aglivc', 'stdedc', 'bglivc', 'somtc']]
        outputs = outputs.assign(sd=sd)
        outputs = outputs.assign(treatment=treatment)
        outputs = outputs.assign(rest_period=rest_period)
        return outputs
    
    df_list = []
    # zero sd
    outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\rest_effect\zero_sd"
    sd = 0
    treatment = 'continuous'
    rest_period = 'NA'
    df_list.append(get_from_century_results(outdir, sd, treatment, rest_period))
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\rest_effect"
    sd_list = [0.25, 0.5, 0.75, 1]
    for sd in sd_list:
        # continuous
        treatment = 'continuous'
        rest_period = 'NA'
        outdir = os.path.join(outer_dir, '{}_cont'.format(sd))
        df_list.append(get_from_century_results(outdir, sd, treatment,
                                                rest_period))
        for rest_period in [1, 2, 3, 4]:
            treatment = 'rest_rotation'
            # rotated
            outdir = os.path.join(outer_dir,
                                  '{}_rest_{}_per_ha'.format(rest_period, sd))
            df_list.append(get_from_century_results(outdir, sd, treatment,
                                                    rest_period))
    rest_result_df = pd.concat(df_list)
    rest_result_df.to_csv(save_as)

def calc_equivalent_intake():
    """Calculate the approximately equivalent amount of one feed type that
    would need to be eaten to replace another. For example, how much hay would
    be nutritionally equivalent to a given amount of forage, in terms of
    liveweight gain?"""
    
    def calc_liveweight(site, herb_class, available_forage):
        supp_available = 0
        DOY = 150
        prop_legume = 0
        herb_class.calc_distance_walked(site.S, herb_class.stocking_density,
                                        available_forage)
        max_intake = herb_class.calc_max_intake()
        ZF = herb_class.calc_ZF()
        HR = forage_u.calc_relative_height(available_forage)
        diet = forage_u.diet_selection_t2(ZF, HR, prop_legume, supp_available, 
                                          max_intake, herb_class.FParam,
                                          available_forage)
        diet_interm = forage_u.calc_diet_intermediates(diet, herb_class,
                                                       prop_legume, DOY, site)
        reduced_max_intake = forage_u.check_max_intake(diet, diet_interm,
                                                       herb_class, max_intake)
        if reduced_max_intake < max_intake:
            diet = forage_u.diet_selection_t2(ZF, HR, prop_legume,
                                              supp_available, 
                                              reduced_max_intake,
                                              herb_class.FParam,
                                              available_forage)
            diet_interm = forage_u.calc_diet_intermediates(diet, herb_class,
                                                           prop_legume, DOY,
                                                           site)
        delta_W = forage_u.calc_delta_weight(diet_interm, herb_class)
        return delta_W
    
    grass_gm2 = 87.  # does it matter?
    forage_args = default_forage_args()
    site = forage_u.SiteInfo(1., 44.6)
    grass_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_inputs\Ucross\grass_high_quality_only.csv"
    hay_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_inputs\Ucross\Merlin_hay.csv" 
    herb_csv = forage_args['herbivore_csv']
    h_class = (pd.read_csv(forage_args[u'herbivore_csv'])).to_dict(
                                                           orient='records')[0]
    herd = forage_u.HerbivoreClass(h_class)
    herd.update()
    
    # get liveweight from grass
    grass_list = (pd.read_csv(grass_csv)).to_dict(orient='records')
    for grass in grass_list:
        grass['green_gm2'] = grass_gm2
        grass['dead_gm2'] = grass_gm2 * 0.5
    available_forage = forage_u.calc_feed_types(grass_list)
    for feed_type in available_forage:
        feed_type.calc_digestibility_from_protein(
                                             forage_args['digestibility_flag'])
    grass_liveweight = calc_liveweight(site, herd, available_forage)
    
    # try n times to match liveweight with hay, given a tolerance, and changing
    # the amount of hay supplied
    hay_list = (pd.read_csv(hay_csv)).to_dict(orient='records')
    tolerance = 10.
    hay_liveweight = 0
    upper_bound = grass_gm2 * 2.
    lower_bound = grass_gm2 / 2.
    n = 1
    while n < 15:
        for hay in hay_list:
            hay['green_gm2'] = upper_bound
            hay['dead_gm2'] = upper_bound * 0.5
        available_forage = forage_u.calc_feed_types(hay_list)
        for feed_type in available_forage:
            feed_type.calc_digestibility_from_protein(
                                             forage_args['digestibility_flag'])
        hay_upper_lw = calc_liveweight(site, herd, available_forage)
        if (grass_liveweight - hay_upper_lw) > tolerance:
            raise Exception, "upper bound hay gm2 is too low"
        for hay in hay_list:
            hay['green_gm2'] = lower_bound
            hay['dead_gm2'] = lower_bound * 0.5
        available_forage = forage_u.calc_feed_types(hay_list)
        for feed_type in available_forage:
            feed_type.calc_digestibility_from_protein(
                                             forage_args['digestibility_flag'])
        hay_lower_lw = calc_liveweight(site, herd, available_forage)
        if (hay_lower_lw - grass_liveweight) > tolerance:
            raise Exception, "lower bound hay gm2 is too high"
        if (hay_upper_lw - grass_liveweight) > (grass_liveweight - hay_lower_lw):
            upper_bound = lower_bound + ((upper_bound - lower_bound) / 2.)
        else:
            lower_bound = lower_bound + ((upper_bound - lower_bound) / 2.)
        n = n + 1
    print "Hay biomass: between {} and {}".format(lower_bound, upper_bound)
    print """liveweight gain: {0:.3f} (grass), {0:.3f} (hay upper bound),
            {0:.3f} (hay lower bound)""".format(grass_liveweight, hay_upper_lw,
                                              hay_lower_lw)
    
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

if __name__ == "__main__":
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\stocking_density_test"
    # stocking_dens_test_wrapper(outer_outdir)
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\WitW\model_results\Ucross\stocking_density_test_force_intake"
    # stocking_dens_test_wrapper(outer_outdir)
    # erase_intermediate_files(outer_outdir)
    composition_wrapper()