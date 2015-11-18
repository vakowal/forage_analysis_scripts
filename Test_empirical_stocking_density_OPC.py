# Test forage model by running it with observed (estimated) stocking densities
# on OPC

import os
import re
import sys
import shutil
from subprocess import Popen
import pandas

sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage
import forage_century_link_utils as cent
import freer_param as FreerParam

def run_simulations():
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    fix_file = 'drytrpfi.100'
    graz_file = os.path.join(century_dir, "graz.100")
    site_list = ['Research', 'Loidien']  #, 'Rongai', 'Kamok']
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Output/Stocking_density_test"
    prop_legume = 0
    template_level = 'GL'
    herb_class_weights = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/Boran_weights.csv"
    sd_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/OPC_stocking_density'
    breed = 'Boran'
    steepness = 1.
    latitude = 0
    supp_available = 0
    FParam = FreerParam.FreerParam(forage.get_general_breed(breed))
    supp = forage.Supplement(FParam, 0, 0, 0, 0, 0, 0)
    forage.set_time_step('month')
    add_event = 1
    
    grass_file = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass.csv"
    grass = (pandas.read_csv(grass_file)).to_dict(orient='records')[0]
    grass['DMD_green'] = 0.64
    grass['DMD_dead'] = 0.64
    grass['cprotein_green'] = 0.1
    grass['cprotein_dead'] = 0.1

    for site in site_list:
        spin_up_outputs = [site+'_hist_log.txt', site+'_hist.lis']
        century_outputs = [site+'_log.txt', site+'.lis', site+'.bin']
        filename = 'average_animals_%s_2km_per_ha.csv' % site
        stocking_density_file = os.path.join(sd_dir, filename)
        sd_df = pandas.read_table(stocking_density_file, sep=',')
        
        outdir = os.path.join(outer_dir, site)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # write CENTURY bat for spin-up simulation
        hist_bat = os.path.join(input_dir, (site + '_hist.bat'))
        hist_schedule = site + '_hist.sch'
        hist_output = site + '_hist'
        cent.write_century_bat(input_dir, hist_bat, hist_schedule,
                               hist_output, fix_file, 'outvars.txt')
        # write CENTURY bat for extend simulation
        extend_bat = os.path.join(input_dir, site + '.bat')
        schedule = site + '.sch'
        output = site
        extend = site + '_hist'
        cent.write_century_bat(century_dir, extend_bat, schedule,
                               output, fix_file, 'outvars.txt', extend)
        # move CENTURY run files to CENTURY dir
        site_file = os.path.join(input_dir, site + '.100')
        weather_file = os.path.join(input_dir, site + '.wth')
        e_schedule = os.path.join(input_dir, site + '.sch')
        h_schedule = os.path.join(input_dir, site + '_hist.sch')
        file_list = [hist_bat, extend_bat, e_schedule, h_schedule, site_file,
                     weather_file]
        for file in file_list:
            shutil.copyfile(file, os.path.join(century_dir,
                                               os.path.basename(file)))
        
        # make a copy of the original graz params and schedule file
        shutil.copyfile(graz_file, os.path.join(century_dir, 'graz_orig.100'))
        label = os.path.basename(e_schedule)[:-4]
        copy_name = label + '_orig.sch'
        shutil.copyfile(e_schedule, os.path.join(input_dir, copy_name))
        
        # run CENTURY for spin-up up to start_year and start_month
        hist_bat = os.path.join(century_dir, site + '_hist.bat')
        century_bat = os.path.join(century_dir, site + '.bat')
        p = Popen(["cmd.exe", "/c " + hist_bat], cwd=century_dir)
        stdout, stderr = p.communicate()
        p = Popen(["cmd.exe", "/c " + century_bat], cwd=century_dir)
        stdout, stderr = p.communicate()
        
        # save copies of CENTURY outputs, but remove from CENTURY dir
        intermediate_dir = os.path.join(outdir, 'CENTURY_outputs_spin_up')
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        to_move = century_outputs + spin_up_outputs
        for file in to_move:
            shutil.copyfile(os.path.join(century_dir, file), os.path.join(
                            intermediate_dir, file))
            os.remove(os.path.join(century_dir, file))
        
        grass_list = [grass]
        results_dict = {'year': [], 'month': []}
        herbivore_input = (pandas.read_csv(herb_class_weights).to_dict(
                           orient='records'))
        herbivore_list = []
        for h_class in herbivore_input:
            results_dict[h_class['label'] + '_gain_kg'] = []
            results_dict[h_class['label'] + '_offtake'] = []
        results_dict['milk_prod_kg'] = []
        for grass in grass_list:
            results_dict[grass['label'] + '_green_kgha'] = []
            results_dict[grass['label'] + '_dead_kgha'] = []
        results_dict['total_offtake'] = []
        results_dict['stocking_density'] = []
        try:
            for row in xrange(len(sd_df)):
                herbivore_list = []
                for h_class in herbivore_input:
                    herd = forage.HerbivoreClass(FParam, breed,
                                                 h_class['weight'],
                                                 h_class['sex'], h_class['age'],
                                                 h_class['stocking_density'],
                                                 h_class['label'], Wbirth=24)
                    herd.update(FParam, 0, 0)
                    herbivore_list.append(herd)
                for h_class in herbivore_list:
                    h_class.stocking_density  = sd_df.iloc[row][h_class.label]
                total_SD = forage.calc_total_stocking_density(herbivore_list)
                results_dict['stocking_density'].append(total_SD)
                siteinfo = forage.SiteInfo(total_SD, steepness, latitude)
                month = sd_df.iloc[row].month
                year = sd_df.iloc[row].year
                suf = '%d-%d' % (month, year)
                DOY = month * 30
                # get biomass and crude protein for each grass type from CENTURY
                output_file = os.path.join(intermediate_dir, site + '.lis')
                outputs = cent.read_CENTURY_outputs(output_file, year, year + 2)
                target_month = cent.find_prev_month(year, month)
                grass['prev_g_gm2'] = grass['green_gm2']
                grass['prev_d_gm2'] = grass['dead_gm2']
                grass['green_gm2'] = outputs.loc[target_month, 'aglivc']
                grass['dead_gm2'] = outputs.loc[target_month, 'stdedc']
                grass['cprotein_green'] = (outputs.loc[target_month,
                                           'aglive1'] / outputs.loc[
                                                       target_month, 'aglivc'])
                grass['cprotein_dead'] = (outputs.loc[target_month,
                                          'stdede1'] / outputs.loc[
                                                       target_month, 'stdedc'])
                if row == 0:
                    available_forage = forage.calc_feed_types(grass_list)
                else:
                    available_forage = forage.update_feed_types(grass_list,
                                                              available_forage)
                results_dict['year'].append(year)
                results_dict['month'].append(month)
                for feed_type in available_forage:
                    results_dict[feed_type.label + '_' +
                                 feed_type.green_or_dead +
                                 '_kgha'].append(feed_type.biomass)

                siteinfo.calc_distance_walked(FParam, available_forage)
                for feed_type in available_forage:
                    feed_type.calc_digestibility_from_protein()
                total_biomass = forage.calc_total_biomass(available_forage)
                # Initialize containers to track forage consumed across herbivore
                # classes
                total_intake_step = 0.
                total_consumed = {}
                for feed_type in available_forage:
                    label_string = ';'.join([feed_type.label,
                                            feed_type.green_or_dead])
                    total_consumed[label_string] = 0.

                for herb_class in herbivore_list:
                    max_intake = herb_class.calc_max_intake(FParam)
                    if herb_class.Z < FParam.CR7:
                        ZF = 1. + (FParam.CR7 - herb_class.Z)
                    else:
                        ZF = 1.
                    if herb_class.stocking_density > 0:
                        adj_forage = forage.calc_adj_availability(
                                                   available_forage,
                                                   herb_class.stocking_density)
                    else:
                        adj_forage = list(available_forage)
                    diet = forage.diet_selection_t2(ZF, prop_legume,
                                                    supp_available, supp,
                                                    max_intake, FParam,
                                                    adj_forage)
                    diet_interm = forage.calc_diet_intermediates(FParam, diet,
                                    supp, herb_class, siteinfo, prop_legume,
                                    DOY)
                    reduced_max_intake = forage.check_max_intake(FParam, diet,
                                                diet_interm, herb_class,
                                                max_intake)
                    if reduced_max_intake < max_intake:
                        diet = forage.diet_selection_t2(ZF, prop_legume,
                                                        supp_available, supp,
                                                        reduced_max_intake, 
                                                        FParam, adj_forage)
                        diet_interm = forage.calc_diet_intermediates(FParam,
                                        diet,
                                        supp, herb_class, siteinfo,
                                        prop_legume, DOY)
                    total_intake_step += (forage.convert_daily_to_step(diet.If)
                                          * herb_class.stocking_density)
                    if herb_class.sex == 'lac_female':
                        milk_production = forage.check_milk_production(FParam,
                                                                   diet_interm)
                        milk_kg_day = forage.calc_milk_yield(FParam,
                                                             milk_production)
                    delta_W = forage.calc_delta_weight(FParam, diet,
                                                       diet_interm,
                                                       supp, herb_class)
                    delta_W_step = forage.convert_daily_to_step(delta_W)
                    herb_class.update(FParam, delta_W_step,
                                      forage.find_days_per_step())
                    if herb_class.stocking_density > 0:
                        results_dict[herb_class.label + '_gain_kg'].append(
                                                                  delta_W_step)
                        results_dict[herb_class.label + '_offtake'].append(
                                                                       diet.If)
                    else:
                        results_dict[herb_class.label + '_gain_kg'].append('NA')
                        results_dict[herb_class.label + '_offtake'].append('NA')
                    if herb_class.sex == 'lac_female':
                        results_dict['milk_prod_kg'].append(milk_kg_day * 30.)

                    # after have performed max intake check, we have the final diet
                    # selected
                    # calculate percent live and dead removed for each grass type
                    consumed_by_class = forage.calc_percent_consumed(
                                        available_forage, diet,
                                        herb_class.stocking_density)
                    forage.sum_percent_consumed(total_consumed,
                                                consumed_by_class)

                results_dict['total_offtake'].append(total_intake_step)
                # send to CENTURY for this month's scheduled grazing event
                date = year + float('%.2f' % (month / 12.))
                schedule = os.path.join(century_dir, site + '.sch')
                target_dict = cent.find_target_month(add_event, schedule, date,
                                                     1)
                if target_dict == 0:
                    er = "Error: no opportunities exist to add grazing event"
                    raise Exception(er)
                new_code = cent.add_new_graz_level(grass, total_consumed,
                                                   graz_file,
                                                   template_level,
                                                   outdir, suf)
                cent.modify_schedule(schedule, add_event, target_dict,
                                     new_code, outdir, suf)

                # call CENTURY from the batch file
                century_bat = os.path.join(century_dir, site + '.bat')
                p = Popen(["cmd.exe", "/c " + century_bat],
                          cwd=century_dir)
                stdout, stderr = p.communicate()
                # save copies of CENTURY outputs, but remove from CENTURY dir
                intermediate_dir = os.path.join(outdir,
                                     'CENTURY_outputs_m%d_y%d' % (month, year))
                if not os.path.exists(intermediate_dir):
                    os.makedirs(intermediate_dir)
                for file in century_outputs:
                    shutil.copyfile(os.path.join(century_dir, file),
                                    os.path.join(intermediate_dir, file))
                    os.remove(os.path.join(century_dir, file))

        # remove files from CENTURY directory
        finally:
            # replace graz params used by CENTURY with original file
            os.remove(graz_file)
            shutil.copyfile(os.path.join(century_dir, 'graz_orig.100'),
                            graz_file)
            os.remove(os.path.join(century_dir, 'graz_orig.100'))
            files_to_remove = [os.path.join(century_dir, os.path.basename(
                                                       f)) for f in file_list]
            for file in files_to_remove:
                os.remove(file)
            os.remove(os.path.join(century_dir, site + '_hist.bin'))
            filled_dict = forage.fill_dict(results_dict, 'NA')
            df = pandas.DataFrame(filled_dict)
            df.to_csv(os.path.join(outdir, 'summary_results.csv'))

def calculate_density(in_folder, result_dir):
    """Convert counts of animals within a certain distance of a point to
    stocking density per ha, which is what the forage model expects as input.
    The distance is expected to be an integer, in km, and the name of data
    files containing animal numbers is expected to follow the form
    'average_animals_<site>_<distance>km'."""
    
    files = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(
                                                                in_folder, f))]
    ab_files = [f for f in files if re.search('^average_animals', f)]
    file = ab_files[0]
    km = int(re.search('_(.)km', file).group(1))
    sq_km = pow((km * 2), 2)
    ha = sq_km * 100
    for file in ab_files:
        df = pandas.read_table(os.path.join(in_folder, file), sep=',')
        df = df.where((pandas.notnull(df)), 0)
        vals = df[['Bulls', 'Calves', 'Cows', 'Heifers', 'Steers', 'Weaners',
                   'steer/heifer']]
        per_ha = vals.div(ha)
        per_ha['month'] = df['month']
        per_ha['year'] = df['year']
        out_name = os.path.join(result_dir, file[:-4] + '_per_ha.csv')
        per_ha.to_csv(out_name)



if __name__ == "__main__":
    in_folder = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Sharon/From_Sharon_5.29.15/Matched_GPS_records/Matched_with_weather_stations'
    result_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/OPC_stocking_density"
    # calculate_density(in_folder, result_dir)
    run_simulations()