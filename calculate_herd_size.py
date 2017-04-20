# how many cattle would be supported by eating a certain amount?
# the amount of feed consumed was calculated by back-calculating management
# to match an empirical biomass target in CENTURY

import math
import os
import pandas as pd
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/rangeland_production')
import forage_utils as forage
import forage_century_link_utils as cent

def regional_properties():
    """Use the forage model to estimate forage intake by an animal at each
    step of a back-calculated schedule. calculate this for 24 months prior to
    empirical measurement date."""
    
    save_as = 'C:/Users/Ginger/Desktop/intake_record.csv'
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014\back_calc_match 2015"
    template_herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\herd_avg_uncalibrated.csv"
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015/0.csv"
    
    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\regional_PDM_summary.csv"
    site_df = pd.read_csv(match_csv)
    site_df = site_df.loc[site_df.Year == 2015]
    result_dict = {'FID': [], 'year': [], 'month': [],
                   'intake_forage_indiv': []}
    forage.set_time_step('month')
    for FID in site_df.FID:
        herbivore_list = []
        herbivore_input = (pd.read_csv(template_herb_csv).to_dict(
                           orient='records'))
        for h_class in herbivore_input:
            herd = forage.HerbivoreClass(h_class)
            herd.update()
            herbivore_list.append(herd)
        stocking_density_dict = forage.populate_sd_dict(herbivore_list)
        total_SD = forage.calc_total_stocking_density(herbivore_list)
        sub_df = site_df.loc[site_df.FID == FID]
        date_list = sub_df.sim_date.values[0].rsplit("/")
        start_year = int(date_list[2]) - 2  # 24 months prior to measurement date
        start_month = int(date_list[0])
        century_date = round((int(start_year) + int(start_month) / 12.0), 2)
        site_dir = os.path.join(outer_outdir, 'FID_{}'.format(FID))
        output_dirs = [f for f in os.listdir(site_dir) if os.path.isdir(
                        os.path.join(site_dir, f))]
        output_dirs = [f for f in os.listdir(site_dir) if f.startswith('CENTURY_outputs_iteration')]
        iter_list = []
        for dir in output_dirs:
            iter_list.extend([int(s) for s in dir.split('n') if s.isdigit()])
        final_out_iter = max(iter_list)
        output_file = os.path.join(site_dir,
                                   'CENTURY_outputs_iteration{}'.format(
                                                               final_out_iter),
                                    '{}.lis'.format(FID))
        outputs = cent.read_CENTURY_outputs(output_file, start_year,
                                            start_year + 3)
        outputs.drop_duplicates(inplace=True)
        grass_list = (pd.read_csv(grass_csv)).to_dict(orient='records')
        for step in xrange(1, 25):  # so that we can use the function 'find_prev_month'
            step_month = start_month + step
            if step_month > 12:
                mod = step_month % 12
                if mod == 0:
                    month = 12
                    year = (step_month / 12) + start_year - 1
                else:
                    month = mod
                    year = (step_month / 12) + start_year
            else:
                month = step_month
                year = (step / 12) + start_year
            target_month = cent.find_prev_month(year, month)
            for grass in grass_list:
                grass['green_gm2'] = outputs.loc[target_month, 'aglivc']
                grass['dead_gm2'] = outputs.loc[target_month, 'stdedc']
            available_forage = forage.calc_feed_types(grass_list)
            for feed_type in available_forage:
                feed_type.calc_digestibility_from_protein()
            diet_dict = {}
            for herb_class in herbivore_list:
                herb_class.calc_distance_walked(1., total_SD,
                                                    available_forage)
                max_intake = herb_class.calc_max_intake()
                ZF = herb_class.calc_ZF()
                HR = forage.calc_relative_height(available_forage)
                diet = forage.diet_selection_t2(ZF, HR, 0., 0, max_intake,
                                                herb_class.FParam,
                                                available_forage,
                                                herb_class.f_w, herb_class.q_w)
                diet_interm = forage.calc_diet_intermediates(diet, herb_class,
                                                             0., 150)
                reduced_max_intake = forage.check_max_intake(diet, diet_interm,
                                                             herb_class,
                                                             max_intake)
                if reduced_max_intake < max_intake:
                    diet = forage.diet_selection_t2(ZF, HR, 0., 0, max_intake,
                                                    herb_class.FParam,
                                                    available_forage,
                                                    herb_class.f_w,
                                                    herb_class.q_w)
                diet_dict[herb_class.label] = diet
            forage.reduce_demand(diet_dict, stocking_density_dict,
                                 available_forage)
            assert len(diet_dict.keys()) == 1
            hclass_label = diet_dict.keys()[0]
            intake = forage.convert_daily_to_step(diet_dict[hclass_label].If)
            result_dict['FID'].append(FID)
            result_dict['year'].append(year)
            result_dict['month'].append(month)
            result_dict['intake_forage_indiv'].append(intake)
    df = pd.DataFrame(result_dict)
    df.to_csv(save_as)

def summarize_intake(summary_csv, save_as):
    """summarize intake within sites, calculated by regional_properties()"""
    
    df = pd.read_csv(summary_csv)
    grouped = df.groupby('FID')
    avg_intake = grouped['intake_forage_indiv'].mean()
    avg_intake.to_csv(save_as)
    
def jenny_sites():
    """Use IPCC tier 2 methods to convert forage eaten to animals"""
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    average_weight_kg = 160.
    time_step = 'month'
    forage.set_time_step(time_step)
    outdir = 'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\modify_management'

    GT_site = {'name': 'GT', 'biomass': 341.1, 'date': 2012.58}
    M05_site = {'name': 'M05', 'biomass': 113.3, 'date': 2013.58}
    MO_site = {'name': 'MO', 'biomass': 181.1, 'date': 2012.58}
    N4_site = {'name': 'N4', 'biomass': 300.0, 'date': 2014.17}
    W3_site = {'name': 'W3', 'biomass': 126.3, 'date': 2014.17}
    W06_site = {'name': 'W06', 'biomass': 322.1, 'date': 2013.58}

    site_list = [W3_site, N4_site]

    for site in site_list:
        empirical_date = site['date']
        schedule = site['name'] + '_mod.sch'
        century_bat = site['name'] + '.bat'
        output = site['name'] + 'mod-manag.lis'
        schedule_file = os.path.join(century_dir, schedule)
        output_file = os.path.join(century_dir, output)
        graz_file = os.path.join(century_dir, 'graz_' + site['name'] + '.100')
        
        first_year = math.floor(empirical_date) - 5
        last_year = math.ceil(empirical_date)
        
        biomass_df = cent.read_CENTURY_outputs(output_file, first_year, last_year)
        g_level = cent.read_graz_level(schedule_file)
        params = cent.read_graz_params(graz_file)
        g_level_rpt = cent.process_graz_level(g_level, params)
        graz_rest = g_level_rpt.loc[(g_level_rpt.index >= first_year), ]
        for forage_quality in ['low', 'moderate', 'high']:
            event_list = []        
            herd_size_list = []
            date_list = []
            for event in graz_rest.index:
                try:            
                    livec = biomass_df.loc[event, 'aglivc']
                except KeyError:
                    print str(event) + ' not in CENTURY output'
                    continue
                date = cent.convert_to_year_month(event)
                date_list.append('%d/%d' % (date[0], date[1]))            
                event_list.append(event)
                stdedc = biomass_df.loc[event, 'stdedc']
                flgrem = graz_rest.loc[event, 'flgrem']
                fdgrem = graz_rest.loc[event, 'fdgrem']
                live_cons = livec * flgrem
                stded_cons = stdedc * fdgrem
                intake_l = cent.convert_units(live_cons, 1)
                intake_d = cent.convert_units(stded_cons, 1)
                MJ_per_kg_DM = forage.calc_energy_t1(forage_quality)
                DMI_per_indiv = forage.calc_DMI_t1(MJ_per_kg_DM, average_weight_kg)
                herd_size = float(intake_l + intake_d) / DMI_per_indiv
                herd_size_list.append(herd_size)
            dict1 = {'date': date_list, 'herd_size': herd_size_list, 'event': event_list}
            herd_df = pd.DataFrame(dict1)
            herd_sort = herd_df.sort(['event'])
            csvfile = os.path.join(outdir, site['name'] + 'herd_size_' + forage_quality + '.csv')
            herd_sort.to_csv(csvfile)
            
if __name__ == "__main__":
    # regional_properties()
    summary_csv = 'C:/Users/Ginger/Desktop/intake_record.csv'
    save_as = 'C:/Users/Ginger/Desktop/intake_avg_per_site.csv'
    summarize_intake(summary_csv, save_as)
    