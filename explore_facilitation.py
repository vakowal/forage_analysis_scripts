## explore potential for different animal types to facilitate each other
## through differential diet selection

import os
import shutil
import pandas
import math
import numpy as np
from operator import attrgetter
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage_u
import freer_param as FreerParam
import forage

def calculate_new_growth(run_dir, grass_csv, year):
    """Calculate the percentage of biomass that is new growth each month.
    run_dir is the directory of the model run. grass_csv is the input csv
    describing grasses used in the simulation. year is the year for which we
    will calculate percent new growth."""
    
    cent_dir = "CENTURY_outputs_m12_y%s" % str(year)
    biomass_t = [0] * 12
    new_growth_t = [0] * 12
    live_biomass_t = [0] * 12
    grass_list = pandas.read_csv(grass_csv)
    for gr in grass_list.label.unique():
        filename = os.path.join(run_dir, cent_dir, "%s.lis" % gr)
        gr_df = pandas.read_fwf(filename)
        sub_df = gr_df.loc[gr_df['time'] > year]
        sub_df = sub_df.loc[sub_df['time'] <= year + 1]
        sub_df = sub_df.drop_duplicates('time')
        growth = sub_df.agcacc.values
        live_biomass = sub_df.aglivc.values
        stded_biomass = sub_df.stdedc.values
        biomass = live_biomass + stded_biomass
        n_content = sub_df['aglive(1)'].values / live_biomass
        new_growth = [growth[0]]
        for idx in range(1,12):
            new_growth.append(growth[idx] - growth[idx - 1])
        new_growth_t = [a + b for a, b in zip(new_growth_t, new_growth)]
        live_biomass_t = [a + b for a, b in zip(live_biomass_t, live_biomass)]
        biomass_t = [a + b for a, b in zip(biomass_t, biomass)]
    perc_new_growth_of_live = [a / b for a, b in zip(new_growth_t,
                                                     live_biomass_t)]
    perc_new_growth_of_total = [a / b for a, b in zip(new_growth_t, biomass_t)]
    perc_live_of_total = [a / b for a, b in zip(live_biomass_t, biomass_t)]
    return new_growth, biomass_t, perc_new_growth_of_live, perc_live_of_total, perc_new_growth_of_total, n_content

def fabricate_forage(total_biomass, abun_ratio, cp_mean, cp_ratio,
                     abun_cp_same):
    """Populate available forage with two grass types that relate to each other
    as described by the abundance ratio (larger:smaller) and the crude protein
    ratio (larger:smaller).  If abun_cp_same is true, the grass with higher
    abundance also has higher crude protein."""
    
    lower_abun = float(total_biomass) / (abun_ratio + 1.)
    higher_abun = float(total_biomass) - lower_abun
    lower_cp = (2. * cp_mean) / (cp_ratio + 1.)
    higher_cp = float(cp_ratio) * lower_cp
    
    available_forage = []
    if abun_cp_same:
        available_forage_u.append(forage_u.FeedType('grass1', 'green', lower_abun,
                                0., lower_cp, 'C4'))
        available_forage_u.append(forage_u.FeedType('grass2', 'green', higher_abun,
                                0., higher_cp, 'C4'))
    else:
        available_forage_u.append(forage_u.FeedType('grass1', 'green', lower_abun,
                                0., higher_cp, 'C4'))
        available_forage_u.append(forage_u.FeedType('grass2', 'green', higher_abun,
                                0., lower_cp, 'C4'))
    for feed_type in available_forage:
        feed_type.rel_availability = feed_type.biomass / total_biomass
        feed_type.calc_digestibility_from_protein()
    return available_forage

def diet_selection_t2(ZF, HR, prop_legume, supp_available, supp, Imax, FParam,
                      available_forage, idx, f_w=0, q_w=0, force_supp=None):
    """Perform diet selection for an individual herbivore, tier 2.  This
    function calculates relative availability, F (including factors like
    pasture height and animal mouth size) and relative ingestibility, RQ 
    including factors like pasture digestibility and proportion legume in the
    sward) to select a preferred mixture of forage types, including supplement
    if offered.  Available forage must be supplied to the function in an
    ordered list such that available_forage[0] is of highest digestibility.

    Returns daily intake of forage (kg; including seeds), daily intake of
    supplement (kg), average dry matter digestibility of forage, and average
    crude protein intake from forage_u."""

    available_forage = sorted(available_forage, reverse=True,
                              key=attrgetter('digestibility'))
    diet_selected = forage_u.Diet()
    if Imax == 0:
        for f_index in range(len(available_forage)):
            f_label = available_forage[f_index].label + ';' +\
                        available_forage[f_index].green_or_dead
            diet_selected.intake[f_label] = 0.
        return diet_selected

    F = list()
    RR = list()
    RT = list()
    HF = list()
    RQ = list()
    R = list()
    R_w = list()
    I = list()
    sum_prev_classes = 0.
    UC = 1.
    supp_selected = 0

    for f_index in range(len(available_forage)):
        RQ.append(1. - FParam.CR3 * (FParam.CR1 - (1. - prop_legume)  # eq 21
                  * available_forage[f_index].SF -
                  available_forage[f_index].digestibility))
        if supp_available:
            if RQ[f_index] <= supp.RQ or force_supp:
                supp_selected = 1
                supp_available = 0
                Fs = min((supp.DMO / Imax) / supp.RQ, UC, FParam.CR11 /
                          supp.M_per_D)  # eq 23
                sum_prev_classes += Fs
                UC = max(0., 1. - sum_prev_classes)
        HF.append(1. - FParam.CR12 + FParam.CR12 * HR[f_index])  # eq 18
        RT.append(1. + FParam.CR5 * math.exp(-(1. + FParam.CR13 *  # eq 17
            available_forage[f_index].rel_availability) *
            (FParam.CR6 * HF[f_index] * ZF * available_forage[f_index].biomass)
            ** 2))
        RR.append(1. - math.exp(-(1. + FParam.CR13 *  # eq 16
            available_forage[f_index].rel_availability) * FParam.CR4 *
            HF[f_index] * ZF * available_forage[f_index].biomass))
        F.append(UC * RR[f_index] * RT[f_index])  # eq 14
        sum_prev_classes += F[f_index]
        UC = max(0., 1. - sum_prev_classes)  # eq 15
    # import pdb; pdb.set_trace()
    for f_index in range(len(available_forage)):
        # original GRAZPLAN formulation
        R.append(F[f_index] * RQ[f_index] * (1. + FParam.CR2 * sum_prev_classes
                ** 2 * prop_legume))  # eq 20
        # weight proportional intake by quantity weight and quality weight
        R_w.append(R[f_index] + F[f_index]*f_w + RQ[f_index]*q_w)
    # rescale weighted proportions to retain original sum of R
    sum_Rw = sum(R_w)
    for f_index in range(len(available_forage)):
        R_w[f_index] = (R_w[f_index] / sum_Rw) * sum(R)
    for f_index in range(len(available_forage)):    
        I.append(Imax * R_w[f_index])  # eq 27
        diet_selected.DMDf += (I[f_index] *
                               available_forage[f_index].digestibility)
        diet_selected.CPIf += (I[f_index] *
                               available_forage[f_index].crude_protein)
        diet_selected.If += I[f_index]

        # stash the amount consumed of each forage type
        f_label = available_forage[f_index].label + ';' +\
                  available_forage[f_index].green_or_dead
        diet_selected.intake[f_label] = I[f_index]

    diet_selected.DMDf = diet_selected.DMDf / diet_selected.If
    if supp_selected:
        Rs = Fs * supp.RQ  # eq 25
        diet_selected.Is = Imax * Rs  # eq 30
    return diet_selected

def different_diets(available_forage, herbivore_csv, weight_1):
    """Create different diets for two hypothetical animal types.  The weight
    ratio describes the ratio of """
    
    time_step = 'month'
    forage_u.set_time_step(time_step)
    total_SD = 2
    prop_legume = 0
    DOY = 100
    supp_available = 0
    site = forage_u.SiteInfo(1., -3.25)
    
    supp_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Shem_et_al_1995_supp.csv"
    supp_list = (pandas.read_csv(supp_csv)).to_dict(orient='records')
    supp_info = supp_list[0]
    supp = forage_u.Supplement(FreerParam.FreerParamCattle('indicus'),
                             supp_info['digestibility'],
                             supp_info['kg_per_day'], supp_info['M_per_d'],
                             supp_info['ether_extract'],
                             supp_info['crude_protein'],
                             supp_info['rumen_degradability'])
    herbivore_input = (pandas.read_csv(herbivore_csv)).to_dict(orient='records')
    herbivore_list = []
    for h_class in herbivore_input:
        herd = forage_u.HerbivoreClass(h_class)
        herd.update()
        herbivore_list.append(herd)
    
    avail_weights = [weight_1, 0]
    qual_weights = [0, weight_1]
    
    diet_dict = {}        
    for idx in xrange(len(herbivore_list)):
        herb_class = herbivore_list[idx]
        f_w = avail_weights[idx]
        rq_w = qual_weights[idx]
        herb_class.calc_distance_walked(total_SD, site.S, available_forage)
        max_intake = herb_class.calc_max_intake()

        ZF = herb_class.calc_ZF()
        HR = forage_u.calc_relative_height(available_forage)
        diet = diet_selection_t2(ZF, HR, prop_legume,
                                        supp_available, supp,
                                        max_intake, herb_class.FParam,
                                        available_forage, idx, f_w, rq_w)
        diet_interm = forage_u.calc_diet_intermediates(
                        diet, supp, herb_class, site,
                        prop_legume, DOY)
        # if herb_class.type != 'hindgut_fermenter':
            # reduced_max_intake = forage_u.check_max_intake(diet,
                                                         # diet_interm,
                                                         # herb_class,
                                                         # max_intake)
            # if reduced_max_intake < max_intake:
                # diet = forage_u.diet_selection_t2(ZF, HR,
                                                # args[u'prop_legume'],
                                                # supp_available, supp,
                                                # reduced_max_intake,
                                                # herb_class.FParam,
                                                # available_forage)
        diet_dict[herb_class.label] = diet
    return diet_dict
    # forage_u.reduce_demand(diet_dict, stocking_density_dict,
                         # available_forage)
    diet_interm = forage_u.calc_diet_intermediates(
                                        diet, supp, herb_class, site,
                                        prop_legume, DOY)
    delta_W = forage_u.calc_delta_weight(diet_interm, herb_class)
    delta_W_step = forage_u.convert_daily_to_step(delta_W)
    herb_class.update(delta_weight=delta_W_step,
                      delta_time=forage_u.find_days_per_step())
    
def run_test(save_as):
    """Change forage characteristics, change herbivore characteristics,
    measure outcomes."""
    
    summary_dict = {'total_biomass': [],
                    'abundance_ratio': [],
                    'cp_mean': [],
                    'cp_ratio': [],
                    'abundance_cp_same': [],
                    'weight_1': [],
                    'segregation': [],
                    }
    herbivore_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herbs_diet_illustration.csv"
    for total_biomass in np.linspace(40, 4000, 10):
        for abun_ratio in [10]:  # np.linspace(1.5, 10, 10):
            for cp_mean in np.linspace(0.04, 0.2, 10):
                for cp_ratio in [10]:  # np.linspace(1.5, 10, 10):
                    for abun_cp_same in [0, 1]:
                        for weight_1 in [0.01, 0.1, 1, 10, 100]:
                            available_forage = fabricate_forage(
                                                 total_biomass, abun_ratio,
                                                 cp_mean, cp_ratio,
                                                 abun_cp_same)
                            diet_dict = different_diets(
                                                 available_forage,
                                                 herbivore_csv, weight_1)
                            segregation = calc_diet_segregation(diet_dict)
                            summary_dict['total_biomass'].append(total_biomass)
                            summary_dict['abundance_ratio'].append(abun_ratio)
                            summary_dict['cp_mean'].append(cp_mean)
                            summary_dict['cp_ratio'].append(cp_ratio)
                            summary_dict['abundance_cp_same'].append(
                                                                  abun_cp_same)
                            summary_dict['weight_1'].append(weight_1)
                            summary_dict['segregation'].append(segregation)
    df = pandas.DataFrame(summary_dict)
    df.to_csv(save_as)

def launch_model(herb_csv, outdir, restart_monthly=0):
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input"
    # 'grass_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/facilitation_exploration/grasses_diet_illustration.csv",
    forage_args = {
        'latitude': 0.02759,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 2015,
        'start_month': 1,
        'num_months': 12,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GH',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'herbivore_csv': herb_csv,
        'grass_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/C_dactylon_T_triandra.csv",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
        'restart_yearly': 0,
        'diet_verbose': 1,
        'restart_monthly': restart_monthly,
    }
    forage.execute(forage_args)

def change_herbivore_weights(herb_csv, weight):
    """Manipulate quantity and quality weights."""
    
    df = pandas.read_csv(herb_csv)
    df = df.set_value(0, 'quant_weight', weight)
    df = df.set_value(0, 'qual_weight', 0)
    df = df.set_value(1, 'quant_weight', 0)
    df = df.set_value(1, 'qual_weight', weight)
    df = df.set_index('label')
    df.to_csv(herb_csv)

def summarize_plant_composition(outer_dir, save_as, weight_list):    
    """Compare plant composition from several model runs.  Does variation in
    diet selection between herbivores drive differences in plant community
    composition?"""
    
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\facilitation_exploration\model_runs\6.21.16"
    herb_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/facilitation_exploration/herbs_diet_illustration.csv"
    weight_list = [0, 1, 10]
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\facilitation_exploration\asymmetric_weight_effects_grass_proportions_6.21.16.csv"
    for weight in weight_list:
        outdir = os.path.join(outer_dir, 'outputs_asymmetric_w%d' % weight)
        change_herbivore_weights(herb_csv, weight)
        launch_model(herb_csv, outdir)
    summary_dict = {'weight': [],
                    'step': [],
                    'g1_prop': [],
                    'g2_prop': [],
                    'diff_abs': [],
                    'diff_proportion': [],
                    }
    for weight in weight_list:
        sum_csv = os.path.join(outer_dir, 'outputs_asymmetric_w%d' % weight,
                               'summary_results.csv')
        df = pandas.read_csv(sum_csv)
        sum_g1 = df.grass_1_green_kgha + df.grass_1_dead_kgha
        sum_g2 = df.grass_2_green_kgha + df.grass_2_dead_kgha
        total_g = sum_g1 + sum_g2
        diff_abs = abs(sum_g1 - sum_g2)
        diff_prop = abs((sum_g1 / total_g) - (sum_g2 / total_g))
        summary_dict['g1_prop'] += (sum_g1 / total_g).tolist()
        summary_dict['g2_prop'] += (sum_g2 / total_g).tolist()
        summary_dict['diff_abs'] += diff_abs.tolist()
        summary_dict['diff_proportion'] += diff_prop.tolist()
        summary_dict['step'] += df.step.tolist()
        summary_dict['weight'] += [weight] * len(df.step)
    df = pandas.DataFrame(summary_dict)
    df.to_csv(save_as)

def stocking_density_percent_new_growth_test():
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\facilitation_exploration\model_runs\sd_new_growth\restart_monthly\cattle_new_growth_perc_live_summary.csv"
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/C_dactylon_T_triandra.csv"
    herb_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\facilitation_exploration\cattle_sd_levels"
    outer_out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\facilitation_exploration\model_runs\sd_new_growth\restart_monthly"
    year = 2015
    sum_dict = {'stocking_density': [], 'month': [], 'label': [],
                'biomass': []}
    for sd in [.1, .5, 0.75, 1, 1.25]:
        sum_dict['month'] = sum_dict['month'] + (range(1,13) * 7)
        sum_dict['stocking_density'] = sum_dict['stocking_density'] + [sd] * 84
        sum_dict['label'] = sum_dict['label'] + ['new_growth'] * 12 + \
                                                ['total_biomass'] * 12 + \
                                                ['perc_new_growth_of_live'] * 12 + \
                                                ['perc_live_of_total'] * 12 + \
                                                ['perc_new_growth_of_total'] * 12 + \
                                                ['n_content_of_live'] * 12 + \
                                                ['liveweight_gain'] * 12
        herb_csv = os.path.join(herb_dir, 'cattle_%s.csv' % str(sd))
        outdir = os.path.join(outer_out_dir, 'cattle_%s' % str(sd))
        if not os.path.exists(outdir):
            launch_model(herb_csv, outdir, restart_monthly=1)
        new_growth, biomass_t, perc_new_growth_of_live, perc_live_of_total, perc_new_growth_of_total, n_content = calculate_new_growth(
                                                       outdir, grass_csv, year)
        weight_df = pandas.read_csv(os.path.join(outdir, 'summary_results.csv'))
        weight_df = weight_df.loc[weight_df['year'] == 2015]
        gain = weight_df.cattle_gain_kg.values.tolist()
        sum_dict['biomass'] = sum_dict['biomass'] + new_growth + biomass_t + \
                                perc_new_growth_of_live + perc_live_of_total + \
                                perc_new_growth_of_total + n_content.tolist() + \
                                gain
        
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as, index=False)
    
    
    
if __name__ == "__main__":
    # input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\facilitation_exploration"
    # outer_out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\facilitation_exploration\model_runs"
    # for herb in ['herb1_0.2', 'herb1_2_0.3', 'herb1_2_0.2']:
        # herb_csv = os.path.join(input_dir, '%s.csv' % herb)
        # outdir = os.path.join(outer_out_dir, '%s_grass1_grass2' % herb)
        # launch_model(herb_csv, outdir)   
    # launch_model_and_summarize_plant_composition()
    stocking_density_percent_new_growth_test()