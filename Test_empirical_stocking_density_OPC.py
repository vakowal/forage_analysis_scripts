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
    site_list = ['kamok', 'loidien', 'research', 'rongai']
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Output/Stocking_density_test"
    prop_legume = 0
    template_level = 'GL'

    grass_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass.csv"
    grass = (pandas.read_csv(args[u'grass_csv'])).to_dict(orient='records')[0]

    for site in site_list:
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
        cent.write_century_bat(args[u'century_dir'], extend_bat, schedule,
                               output, fix_file, 'outvars.txt', extend)
        # move batch files to century dir
        schedule = os.path.join(input_dir, site + '.sch')
        files_to_move = [hist_bat, extend_bat, schedule]
        file_list = [os.path.join(input_dir, f) for f in files_to_move]
        for file in file_list:
            shutil.copyfile(file, os.path.join(century_dir, os.path.basename(file)))
        
        # make a copy of the original graz params and schedule file
        shutil.copyfile(graz_file, os.path.join(args[u'century_dir'],
                        'graz_orig.100'))
        label = os.path.basename(schedule)[:-4]
        copy_name = label + '_orig.sch'
        shutil.copyfile(schedule, os.path.join(input_dir, copy_name))
        
        # run CENTURY for spin-up up to start_year and start_month
        hist_bat = os.path.join(args[u'century_dir'], site + '_hist.bat')
        century_bat = os.path.join(args[u'century_dir'], site + '.bat')
        p = Popen(["cmd.exe", "/c " + hist_bat], cwd=args[u'century_dir'])
        stdout, stderr = p.communicate()
        p = Popen(["cmd.exe", "/c " + century_bat], cwd=args[u'century_dir'])
        stdout, stderr = p.communicate()
        
        grass_list = [grass]
        # stocking_density_file = ?
        sd_df = pandas.read_table(stocking_density_file, sep=',')
        results_dict = {'year': [], 'month': []}
        for h_class in ['Bulls', 'Cows', 'Calves', 'Heifers', 'Steers', 'Weaners',
                        'steer/heifer']:
            results_dict[h_class + '_gain_kg'] = []
        results_dict['milk_prod_kg'] = []
        for grass in grass_list:
            results_dict[grass['label'] + '_green_kgha'] = []
            results_dict[grass['label'] + '_dead_kgha'] = []
        
        try:
            for row in xrange(len(sd_df)):
                # TODO populate herbivore_list from sd_df
                total_SD = forage.calc_total_stocking_density(herbivore_list)
                site = forage.SiteInfo(total_SD, args[u'steepness'],
                                       args[u'latitude'])
                month = sd_df.iloc[row].month
                year = sd.df.iloc[row].year
                DOY = month * 30
                # get biomass and crude protein for each grass type from CENTURY
                output_file = os.path.join(args[u'century_dir'], site + '.lis')
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
                    results_dict[feed_type.label + '_' + feed_type.green_or_dead +
                                 '_kgha'].append(feed_type.biomass)

                site.calc_distance_walked(FParam, available_forage)
                for feed_type in available_forage:
                    feed_type.calc_digestibility_from_protein()

                # Initialize containers to track forage consumed across herbivore
                # classes
                total_consumed = {}
                for feed_type in available_forage:
                    label_string = ';'.join([feed_type.label,
                                            feed_type.green_or_dead])
                    total_consumed[label_string] = 0.

                # TODO herb class ordering ('who eats first') goes here
                for herb_class in herbivore_list:
                    max_intake = herb_class.calc_max_intake(FParam)

                    if herb_class.Z < FParam.CR7:
                        ZF = 1. + (FParam.CR7 - herb_class.Z)
                    else:
                        ZF = 1.

                    diet = forage.diet_selection_t2(ZF, prop_legume,
                                                    supp_available, supp,
                                                    max_intake, FParam,
                                                    available_forage)
                    diet_interm = forage.calc_diet_intermediates(FParam, diet,
                                    supp, herb_class, site, prop_legume,
                                    DOY)
                    reduced_max_intake = forage.check_max_intake(FParam, diet,
                                                diet_interm, herb_class,
                                                max_intake)
                    if reduced_max_intake < max_intake:
                        diet = forage.diet_selection_t2(ZF, prop_legume,
                                                        supp_available, supp,
                                                        reduced_max_intake, FParam,
                                                        available_forage)
                        diet_interm = forage.calc_diet_intermediates(FParam, diet,
                                        supp, herb_class, site,
                                        prop_legume, DOY)
                    if herb_class.sex == 'lac_female':
                        milk_production = forage.check_milk_production(FParam,
                                                                       diet_interm)
                        milk_kg_day = forage.calc_milk_yield(FParam,
                                                             milk_production)
                    delta_W = forage.calc_delta_weight(FParam, diet, diet_interm,
                                                       supp, herb_class)
                    delta_W_step = forage.convert_daily_to_step(delta_W)
                    herb_class.update(FParam, delta_W_step,
                                      forage.find_days_per_step())
                    results_dict[herb_class.label + '_gain_kg'].append(
                                                                      delta_W_step)
                    if herb_class.sex == 'lac_female':
                        results_dict['milk_prod_kg'].append(milk_kg_day * 30.)

                    # after have performed max intake check, we have the final diet
                    # selected
                    # calculate percent live and dead removed for each grass type
                    consumed_by_class = forage.calc_percent_consumed(
                                        available_forage, diet,
                                        herb_class.stocking_density)
                    forage.sum_percent_consumed(total_consumed, consumed_by_class)

                # send to CENTURY for this month's scheduled grazing event
                date = year + float('%.2f' % (month / 12.))
                schedule = os.path.join(century_dir, site + '.sch')
                target_dict = cent.find_target_month(add_event, schedule, date,
                                                     1)
                new_code = cent.add_new_graz_level(grass, total_consumed,
                                                   graz_file,
                                                   template_level,
                                                   outdir, step)
                cent.modify_schedule(schedule, add_event, target_dict,
                                     new_code, outdir, step)

                # call CENTURY from the batch file
                century_bat = os.path.join(century_dir, site + '.bat')
                p = Popen(["cmd.exe", "/c " + century_bat],
                          cwd=args[u'century_dir'])
                stdout, stderr = p.communicate()

        # remove files from CENTURY directory
        finally:
            # replace graz params used by CENTURY with original file
            os.remove(graz_file)
            shutil.copyfile(os.path.join(args[u'century_dir'], 'graz_orig.100'),
                            graz_file)
            os.remove(os.path.join(args[u'century_dir'], 'graz_orig.100'))
            files_to_remove = [os.path.join(century_dir, f) for f in files_to_move]
            for file in files_to_remove:
                os.remove(file)
            # for grass in grass_list:
                # os.remove(??)
            os.remove(output_file)    

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
    result_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/CENTURY_runs_Kenya/OPC_stocking_density"
    calculate_density(in_folder, result_dir)