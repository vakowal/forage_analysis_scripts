# weight gain consequences of quality differences detected on Kenya Ticks project properties

import os
import pandas
from datetime import datetime
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage
import freer_param as FreerParam

def launch_model(herb_csv, grass_list, outdir):
    f_args = {
        'latitude': 0.02759,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 2015,
        'start_month': 1,
        'num_months': 1,
        'mgmt_threshold': 0.,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': outdir,
        'template_level': 'GH',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 1,
        'user_define_digestibility': 0,
        'herbivore_csv': herb_csv,
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'restart_yearly': 0,
        'diet_verbose': 1,
    }
    now_str = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
    if not os.path.exists(f_args['outdir']):
        os.makedirs(f_args['outdir'])
    forage.write_inputs_log(f_args, now_str)
    forage.set_time_step('month')  # current default, enforced by CENTURY
    add_event = 1  # TODO should this ever be 0?
    steps_per_year = forage.find_steps_per_year()
    if f_args['diet_verbose']:
        master_diet_dict = {}
        diet_segregation_dict = {'step': [], 'segregation': []}
    herbivore_list = []
    if f_args[u'herbivore_csv'] is not None:
        herbivore_input = (pandas.read_csv(f_args[u'herbivore_csv'])).to_dict(
                           orient='records')
        for h_class in herbivore_input:
            herd = forage.HerbivoreClass(h_class)
            herd.update()
            BC = 1  # TODO get optional BC from user
            # if BC:
                # herd.check_BC(BC)
            
            herbivore_list.append(herd)
    results_dict = {'step': [], 'year': [], 'month': []}
    for h_class in herbivore_list:
        results_dict[h_class.label + '_kg'] = []
        results_dict[h_class.label + '_gain_kg'] = []
        results_dict[h_class.label + '_intake_forage_per_indiv_kg'] = []
        if h_class.sex == 'lac_female':
            results_dict['milk_prod_kg'] = []
    results_dict['total_offtake'] = []
    supp_available = 0
    if 'supp_csv' in f_args.keys():
        supp_list = (pandas.read_csv(f_args[u'supp_csv'])).to_dict(
            orient='records')
        assert len(supp_list) == 1, "Only one supplement type is allowed"
        supp_info = supp_list[0]
        supp = forage.Supplement(FreerParam.FreerParamCattle('indicus'),
                                 supp_info['digestibility'],
                                 supp_info['kg_per_day'], supp_info['M_per_d'],
                                 supp_info['ether_extract'],
                                 supp_info['crude_protein'],
                                 supp_info['rumen_degradability'])
        if supp.DMO > 0.:
            supp_available = 1

    stocking_density_dict = forage.populate_sd_dict(herbivore_list)
    total_SD = forage.calc_total_stocking_density(herbivore_list)
    site = forage.SiteInfo(f_args[u'steepness'], f_args[u'latitude'])
    threshold_exceeded = 0
    try:
        for step in xrange(f_args[u'num_months']):
            step_month = f_args[u'start_month'] + step
            if step_month > 12:
                mod = step_month % 12
                if mod == 0:
                    month = 12
                else:
                    month = mod
            else:
                month = step_month
            year = (step / 12) + f_args[u'start_year']
            if month == 1 and f_args['restart_yearly'] and \
                                            f_args[u'herbivore_csv'] is not None:
                threshold_exceeded = 0
                herbivore_list = []
                for h_class in herbivore_input:
                    herd = forage.HerbivoreClass(h_class)
                    herd.update()
                    herbivore_list.append(herd)
            # get biomass and crude protein for each grass type
            available_forage = forage.calc_feed_types(grass_list)
            results_dict['step'].append(step)
            results_dict['year'].append(year)
            results_dict['month'].append(month)
            if not f_args[u'user_define_digestibility']:
                for feed_type in available_forage:
                    feed_type.calc_digestibility_from_protein()
            total_biomass = forage.calc_total_biomass(available_forage)
            if step == 0:
                # threshold biomass, amount of biomass required to be left
                # standing (kg per ha)
                threshold_biomass = total_biomass * float(
                                    f_args[u'mgmt_threshold'])
            diet_dict = {}        
            for herb_class in herbivore_list:
                herb_class.calc_distance_walked(total_SD, site.S,
                                                available_forage)
                max_intake = herb_class.calc_max_intake()

                ZF = herb_class.calc_ZF()
                HR = forage.calc_relative_height(available_forage)
                diet = forage.diet_selection_t2(ZF, HR, f_args[u'prop_legume'],
                                                supp_available, supp,
                                                max_intake, herb_class.FParam,
                                                available_forage,
                                                herb_class.f_w, herb_class.q_w)
                diet_interm = forage.calc_diet_intermediates(
                                diet, supp, herb_class, site,
                                f_args[u'prop_legume'], f_args[u'DOY'])
                if herb_class.type != 'hindgut_fermenter':
                    reduced_max_intake = forage.check_max_intake(diet,
                                                                 diet_interm,
                                                                 herb_class,
                                                                 max_intake)
                    if reduced_max_intake < max_intake:
                        diet = forage.diet_selection_t2(ZF, HR,
                                                        f_args[u'prop_legume'],
                                                        supp_available, supp,
                                                        reduced_max_intake,
                                                        herb_class.FParam,
                                                        available_forage)
                diet_dict[herb_class.label] = diet
            forage.reduce_demand(diet_dict, stocking_density_dict,
                                 available_forage)
            if f_args['diet_verbose']:
                # save diet_dict across steps to be written out later
                master_diet_dict[step] = diet_dict
                # diet_segregation = forage.calc_diet_segregation(diet_dict)
                # diet_segregation_dict['step'].append(step)
                # diet_segregation_dict['segregation'].append(diet_segregation)
            total_intake_step = forage.calc_total_intake(diet_dict,
                                                         stocking_density_dict)
            if (total_biomass - total_intake_step) < threshold_biomass:
                print "Forage consumed violates management threshold"
                threshold_exceeded = 1
                total_intake_step = 0
            for herb_class in herbivore_list:
                if threshold_exceeded:
                    diet_dict[herb_class.label] = forage.Diet()
                diet = diet_dict[herb_class.label]
                # if herb_class.type != 'hindgut_fermenter':
                diet_interm = forage.calc_diet_intermediates(
                                        diet, supp, herb_class, site,
                                        f_args[u'prop_legume'], f_args[u'DOY'])
                if herb_class.sex == 'lac_female':
                    milk_production = forage.check_milk_production(
                                                         herb_class.FParam,
                                                         diet_interm)
                    milk_kg_day = herb_class.calc_milk_yield(
                                                           milk_production)
                if threshold_exceeded:
                    delta_W = -(forage.convert_step_to_daily(herb_class.W))
                else:
                    delta_W = forage.calc_delta_weight(diet_interm,
                                                       herb_class)
                delta_W_step = forage.convert_daily_to_step(delta_W)
                herb_class.update(delta_weight=delta_W_step,
                                  delta_time=forage.find_days_per_step())

                results_dict[herb_class.label + '_kg'].append(herb_class.W)
                results_dict[herb_class.label + '_gain_kg'].append(
                                                                  delta_W_step)
                results_dict[herb_class.label +
                             '_intake_forage_per_indiv_kg'].append(
                                         forage.convert_daily_to_step(diet.If))
                if herb_class.sex == 'lac_female':
                    results_dict['milk_prod_kg'].append(
                                     forage.convert_daily_to_step(milk_kg_day))
            results_dict['total_offtake'].append(total_intake_step)
    finally:
        if f_args['diet_verbose']:
            # df = pandas.DataFrame(diet_segregation_dict)
            # save_as = os.path.join(f_args['outdir'], 'diet_segregation.csv')
            # df.to_csv(save_as, index=False)
            for h_label in master_diet_dict[0].keys():
                new_dict = {}
                new_dict['step'] = master_diet_dict.keys()
                new_dict['DMDf'] = [master_diet_dict[step][h_label].DMDf for
                                    step in master_diet_dict.keys()]
                new_dict['CPIf'] = [master_diet_dict[step][h_label].CPIf for
                                    step in master_diet_dict.keys()]
                grass_labels = master_diet_dict[0][h_label].intake.keys()
                for g_label in grass_labels:
                    new_dict['intake_' + g_label] = \
                          [master_diet_dict[step][h_label].intake[g_label] for
                           step in master_diet_dict.keys()]
                df = pandas.DataFrame(new_dict)
                save_as = os.path.join(f_args['outdir'], h_label + '_diet.csv')
                df.to_csv(save_as, index=False)
        filled_dict = forage.fill_dict(results_dict, 'NA')
        df = pandas.DataFrame(filled_dict)
        df.to_csv(os.path.join(f_args['outdir'], 'summary_results.csv'))

def change_sd(herb_csv, sd):
    """Manipulate quantity and quality weights."""
    
    df = pandas.read_csv(herb_csv)
    df = df.set_value(0, 'stocking_density', sd)
    df = df.set_index('label')
    df.to_csv(herb_csv)

def run_test(sd_range):
    herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\cattle_8.2.16.csv"
    qual_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Felicia\GreenBrownSummary 20160801.csv"
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\weight_gain_consequences\raw_outputs\Dequal"
    qual_df = pandas.read_csv(qual_csv)
    grass_list = [{'prev_d_gm2': 0., 'prev_g_gm2': 0., 'cprotein_dead': 0.0609,
                  'cprotein_green': 0.1473, 'type': 'C4',
                  'percent_biomass': 1., 'label': 'NA', 'DMD_green': 0.,
                  'DMD_dead': 0.}]
    for row in xrange(len(qual_df)):
        year = qual_df.iloc[row].Year
        category = qual_df.iloc[row].Ecological_Classification
        grass_list[0]['dead_gm2'] = qual_df.iloc[row].Brown_biomass
        grass_list[0]['green_gm2'] = qual_df.iloc[row].Green_biomass
        grass_list[0]['prev_d_gm2'] = qual_df.iloc[row].Brown_biomass 
        grass_list[0]['prev_g_gm2'] = qual_df.iloc[row].Brown_biomass 
        for sd in sd_range:
            change_sd(herb_csv, sd)
            outdir = os.path.join(outer_dir, '%s_%d_sd_%f' % (category, year,
                                                              sd))
            launch_model(herb_csv, grass_list, outdir)

def summarize_results():
    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\weight_gain_consequences\raw_outputs\Dequal"
    sum_dict = {'sd': [], 'treatment': [], 'year': [], 'gain_kg_one_month': []}
    folders = [f for f in os.listdir(outer_dir) if os.path.isdir(
                                                   os.path.join(outer_dir, f))]
    for folder in folders:
        sum_dict['sd'].append(folder.split('_')[3])
        sum_dict['treatment'].append(folder.split('_')[0])
        sum_dict['year'].append(folder.split('_')[1])
        sum_csv = os.path.join(outer_dir, folder, 'summary_results.csv')
        sum_df = pandas.read_csv(sum_csv)
        sum_dict['gain_kg_one_month'].append(sum_df.iloc[0].cattle_gain_kg)
    df = pandas.DataFrame(sum_dict)
    df.to_csv(os.path.join(outer_dir, 'gain_summary.csv'))
        
if __name__ == "__main__":
    sd_range = [0.18]  # [0.08, 0.18, 2.]
    run_test(sd_range)
    summarize_results()