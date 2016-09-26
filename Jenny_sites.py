# generate input files for CENTURY simulations at Jenny's sites
# then back-calculate management regime and replicate caging

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas
import back_calculate_management as backcalc
sys.path.append(
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_century_link_utils as cent

def split_hist_sch():
    def make_hist(source_file, hist_file):
        flag = ', exclosure'
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as hist:
            with open(source_file, 'rb') as source:
                line = source.next()
                while flag not in line:
                    if '2014          Last year' in line:
                        newline = '1997          Last year\r\n'
                    else:
                        newline = line
                    hist.write(newline)
                    line = source.next()
        shutil.copyfile(abs_path, hist_file)
        os.remove(abs_path)
    
    def make_extend(source_file, extend_file, weather_file):
        flag = '2             Block # empirical weather'
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as extend:
            extend.write('2010          Starting year\r\n')
            extend.write('2014          Last year\r\n')
            with open(source_file, 'rb') as source:
                next(source)
                next(source)
                line = source.next()
                while 'Year Month Option' not in line:
                    extend.write(line)
                    line = source.next()
                while ', exclosure' not in line:
                    line = source.next()
                newline = 'Year Month Option\r\n'
                extend.write(newline)
                newline = '1             Block # empirical weather, exclosure\r\n'
                extend.write(newline)
                line = source.next()
                while 'Weather choice' not in line:
                    extend.write(line)
                    line = source.next()
                newline = 'F             Weather choice\r\n'
                extend.write(newline)
                newline = '{}\r\n'.format(weather_file)
                extend.write(newline)
                line = source.next()
                while '-999 -999' not in line:
                    extend.write(line)
                    line = source.next()
                extend.write(line)
        shutil.copyfile(abs_path, extend_file)
        os.remove(abs_path)
    
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\jenny_site_summary_open.csv"
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\Jenny_sites"
    site_df = pandas.read_csv(site_csv)
    for row in xrange(len(site_df)):
        site_name = site_df.iloc[row].site
        weather_stn = '{}.wth'.format(site_df.iloc[row].closest_weather)
        e_sch = os.path.join(input_dir, '%s.sch' % site_name)
        orig_sch = os.path.join(input_dir, '%s_orig.sch' % site_name)
        hist_file = os.path.join(input_dir, '%s_hist.sch' % site_name)
        # make_hist(orig_sch, hist_file)
        make_extend(orig_sch, e_sch, weather_stn)
    
def PDM_to_g_m2(PDM):
    """Convert PDM measurement to biomass (grams per square m) following
    Jenny's recommended regression for OPC."""
    
    kg_ha = 332.35 * float(PDM) + 15.857
    gm2 = kg_ha / 10
    return gm2
  
def edit_site_file(template, inputs_dict, save_as):
    fh, abs_path = mkstemp()
    os.close(fh)
    with open(abs_path, 'wb') as newfile:
        first_line = '%s (generated by script)\r\n' % inputs_dict['site_name']
        newfile.write(first_line)
        with open(template, 'rb') as old_file:
            next(old_file)
            for line in old_file:
                if '  \'SITLAT' in line:
                    item = '{:0.12f}'.format(inputs_dict['latitude'])[:7]
                    newline = '%s           \'SITLAT\'\r\n' % item
                elif '  \'SITLNG' in line:
                    item = '{:0.12f}'.format(inputs_dict['longitude'])[:7]
                    newline = '%s           \'SITLNG\'\r\n' % item
                elif '  \'SAND' in line:
                    num = inputs_dict['sand'] / 100.0
                    item = '{:0.12f}'.format(num)[:7]
                    newline = '%s           \'SAND\'\r\n' % item
                elif '  \'SILT' in line:
                    num = inputs_dict['silt'] / 100.0
                    item = '{:0.12f}'.format(num)[:7]
                    newline = '%s           \'SILT\'\r\n' % item
                elif '  \'CLAY' in line:
                    num = inputs_dict['clay'] / 100.0
                    item = '{:0.12f}'.format(num)[:7]
                    newline = '%s           \'CLAY\'\r\n' % item
                elif '  \'BULKD' in line:
                    item = '{:0.12f}'.format(inputs_dict['bldens'])[:7]
                    newline = '%s           \'BULKD\'\r\n' % item
                elif '  \'PH' in line:
                    item = '{:0.12f}'.format(inputs_dict['ph'])[:7]
                    newline = '%s           \'PH\'\r\n' % item
                else:
                    newline = line
                newfile.write(newline)
    shutil.copyfile(abs_path, save_as)
    os.remove(abs_path)
    # generate weather statistics (manually :( )

def edit_sch_file(template, site_name, weather_file, save_as):
    fh, abs_path = mkstemp()
    os.close(fh)
    with open(abs_path, 'wb') as newfile:
        with open(template, 'rb') as old_file:
            for line in old_file:
                if '  Site file name' in line:
                    item = '{:14}'.format('{}.100'.format(site_name))
                    newline = '{}Site file name\r\n'.format(item)
                elif '.wth' in line:
                    newline = '{}\r\n'.format(weather_file)
                else:
                    newline = line
                newfile.write(newline)
    shutil.copyfile(abs_path, save_as)
    os.remove(abs_path)

def remove_grazing_after_empirical_date():
    """Remove all scheduled grazing after the empirical measurement date, to
    compare CENTURY outputs with "ungrazed" grass growth"""
    # I ended up doing this manually. :( but don't forget it needs to be done
    # if you ever regenerate schedule files for these sites.
    pass
    
def generate_inputs():
    date_dict = {2012: 2012.50, 2013: 2013.50, 2014: 2014.08}
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\jenny_site_summary_open.csv"
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\Jenny_sites"
    template_site = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\M05.100"
    template_schedule = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\M05.sch"
    
    site_list = []
    site_df = pandas.read_csv(site_csv)
    for row in xrange(len(site_df)):
        inputs_dict = {}
        inputs_dict['site_name'] = site_df.iloc[row].site
        inputs_dict['latitude'] = site_df.iloc[row].POINT_Y
        inputs_dict['longitude'] = site_df.iloc[row].POINT_X
        inputs_dict['sand'] = site_df.iloc[row].geonode_sn
        inputs_dict['silt'] = site_df.iloc[row].geonode_sl
        inputs_dict['clay'] = site_df.iloc[row].geonode_cl
        inputs_dict['bldens'] = site_df.iloc[row].geonode_bl
        inputs_dict['ph'] = site_df.iloc[row].geonode_ph
        
        site_filename = os.path.join(input_dir, '%s.100' %
                                     inputs_dict['site_name'])
        # edit_site_file(template_site, inputs_dict, site_filename)
        weather_stn = '{}.wth'.format(site_df.iloc[row].closest_weather)
        sch_filename = os.path.join(input_dir, '%s.sch' % 
                                    inputs_dict['site_name'])
        # edit_sch_file(template_schedule, inputs_dict['site_name'], weather_stn,
                      # sch_filename)
        
        year = site_df.iloc[row].year
        empirical_date = date_dict[year]
        empirical_biomass = PDM_to_g_m2(site_df.iloc[row].week0cage)
        
        site_list.append({'name': inputs_dict['site_name'],
                          'biomass': empirical_biomass,
                          'date': empirical_date})
    return site_list    

def summarize_calc_schedules(save_as):
    """summarize the grazing schedules that were calculated via the back-calc
    management regime"""
    
    n_years = 2
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\Comparisons_with_CENTURY\back_calc_mgmt_9.13.16"
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\jenny_site_summary_open.csv"
    date_dict = {2012: 2012.50, 2013: 2013.50, 2014: 2014.08}
    starting_sch_dict = {2012: 14, 2013: 14, 2014: 12}  # number months with grazing scheduled initially
    total_mos_dict = {2012: 30, 2013: 30, 2014: 25}

    sum_dict = {'site': [], 'num_months': [], 'num_grazing_months': [],
                'added_or_subtracted': [], 'succeeded': [],
                'perc_live_removed': [], 'perc_dead_removed': []}
    site_df = pandas.read_csv(site_csv)
    for row in xrange(len(site_df)):
        site_name = site_df.iloc[row].site
        site_dir = os.path.join(outerdir, site_name)
        result_csv = os.path.join(site_dir,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pandas.read_csv(result_csv)
        diff = res_df.iloc[len(res_df) - 1].Simulated_biomass - \
                           res_df.iloc[len(res_df) - 1].Empirical_biomass
        if abs(float(diff)) <= 15.0:
            sum_dict['succeeded'].append('succeeded')
        else:
            sum_dict['succeeded'].append('failed')
        year = site_df.iloc[row].year
        total_mos = total_mos_dict[year]
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [int(re.search('{}_{}(.+?).sch'.format(site_name,
                         site_name), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:
            sum_dict['added_or_subtracted'].append('same')
            sum_dict['site'].append(site_name)
            sum_dict['num_months'].append(total_mos)
            sum_dict['num_grazing_months'].append(total_mos)
            sum_dict['perc_live_removed'].append(0.1)
            sum_dict['perc_dead_removed'].append(0.05)
            continue
        empirical_date = date_dict[year]
        final_sch_iter = max(sch_iter_list)
        final_sch = os.path.join(site_dir, '{}_{}{}.sch'.format(site_name,
                                 site_name, final_sch_iter))
        
        # read schedule file, collect months where grazing was scheduled
        schedule_df = cent.read_block_schedule(final_sch)
        for i in range(0, schedule_df.shape[0]):
            start_year = schedule_df.loc[i, 'block_start_year']
            last_year = schedule_df.loc[i, 'block_end_year']
            if empirical_date > start_year and empirical_date <= last_year:
                break
        relative_empirical_year = int(math.floor(empirical_date) -
                                      start_year + 1)
        empirical_month = int(round((empirical_date - float(math.floor(
                                empirical_date))) * 12))
        
        # find months where grazing took place prior to empirical date
        graz_schedule = cent.read_graz_level(final_sch)
        block = graz_schedule.loc[(graz_schedule["block_end_year"] == 
                                  last_year), ['relative_year', 'month',
                                  'grazing_level']]
        empirical_year = block.loc[(block['relative_year'] ==
                                   relative_empirical_year), ]
        empirical_year = empirical_year.loc[(empirical_year['month'] <=
                                            empirical_month), ]
        prev_year = block.loc[(block['relative_year'] <
                              relative_empirical_year), ]
        prev_year = prev_year.loc[(prev_year['relative_year'] >=
                                  (relative_empirical_year - n_years)), ]
        history = prev_year.append(empirical_year)
        filled_history = cent.fill_schedule(history, relative_empirical_year -
                                            n_years, relative_empirical_year,
                                            empirical_month)
        num_graz_mos = len(filled_history.loc[(filled_history[
                                              'grazing_level'] != 'none'), ])
        if num_graz_mos > starting_sch_dict[year]:
            sum_dict['added_or_subtracted'].append('added')
        elif num_graz_mos < starting_sch_dict[year]:
            sum_dict['added_or_subtracted'].append('subtracted')
        sum_dict['site'].append(site_name)
        sum_dict['num_months'].append(total_mos)
        sum_dict['num_grazing_months'].append(num_graz_mos)
        flgrem = 0.1
        fdgrem = 0.05
        if num_graz_mos > starting_sch_dict[year]:
            grz_files = [f for f in os.listdir(site_dir) if
                         f.startswith('graz')]
            if len(grz_files) > 0:
                grz_iter_list = [int(re.search('graz_{}(.+?).100'.format(
                                 site_name), f).group(1)) for f in grz_files]
                final_iter = max(grz_iter_list)
                final_grz = os.path.join(site_dir, 'graz_{}{}.100'.format(
                                         site_name, final_iter))
                with open(final_grz, 'rb') as old_file:
                    for line in old_file:
                        if 'GL    ' in line:
                            line = old_file.next()
                            if 'FLGREM' in line:
                                flgrem = line[:8].strip()
                                line = old_file.next()
                            if 'FDGREM' in line:
                                fdgrem = line[:8].strip()
                            else:
                                er = "Error: FLGREM expected"
                                raise Exception(er)
        sum_dict['perc_live_removed'].append(flgrem)
        sum_dict['perc_dead_removed'].append(fdgrem)
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as)
        
def collect_results(save_as):
    outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\Comparisons_with_CENTURY\back_calc_mgmt_9.13.16"
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\jenny_site_summary_open.csv"
    date_dict = {2012: 2012.50, 2013: 2013.50, 2014: 2014.08}
    
    succeeded = 0
    failed = []
    sum_dict = {'site': [], 'date': [], 'biomass': [], 'sim_vs_emp': []}
    site_df = pandas.read_csv(site_csv)
    for row in xrange(len(site_df)):
        year = site_df.iloc[row].year
        emp_series = [PDM_to_g_m2(site_df.iloc[row]['week{}cage'.format(week)])
                      for week in [0, 3, 6, 9]]
        emp_dates = [date_dict[year] + x for x in [0, 0.0625, 0.125, 0.1875]]
        site_name = site_df.iloc[row].site
        result_csv = os.path.join(outerdir, site_name,
                          'modify_management_summary_{}.csv'.format(site_name))
        res_df = pandas.read_csv(result_csv)
        diff = res_df.iloc[len(res_df) - 1].Simulated_biomass - \
                           res_df.iloc[len(res_df) - 1].Empirical_biomass
        if abs(float(diff)) <= 15.0:
            succeeded += 1
        else:
            failed.append(site_name)
        final_sim = int(res_df.iloc[len(res_df) - 1].Iteration)
        output_file = os.path.join(outerdir, site_name,
                               'CENTURY_outputs_iteration{}'.format(final_sim),
                               '{}.lis'.format(site_name))
        biomass_df = cent.read_CENTURY_outputs(output_file, year - 1, year + 1)
        sim_months = [date_dict[year] + x for x in [0, 0.08, 0.17]]
        if year == 2014:
            sim_months[1] = 2014.17
        sim_dat = biomass_df.loc[sim_months]
        sim_series = sim_dat.aglivc + sim_dat.stdedc
        
        sum_dict['site'].extend([site_name] * 7)
        sum_dict['biomass'].extend(emp_series)
        sum_dict['sim_vs_emp'].extend(['empirical'] * 4)
        sum_dict['date'].extend(emp_dates)
        sum_dict['biomass'].extend(sim_series)
        sum_dict['sim_vs_emp'].extend(['simulated'] * 3)
        sum_dict['date'].extend(sim_series.index)
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as)
    print "{} sites succeeded".format(succeeded)
    print "these sites failed: {}".format(failed)

def calc_management():
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\Jenny_sites"
    n_years = 2  # how many years to potentially manipulate?
    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Jenny\Comparisons_with_CENTURY\back_calc_mgmt_9.13.16"  # where to put results of this routine
    vary = 'both' # 'schedule', 'intensity', 'both'
    live_or_total = 'total'  # 'live' (live biomass) or 'total' (live + standing dead biomass)
    threshold = 15.0  # must match biomass within this many g per sq m
    max_iterations = 40  # number of times to try
    fix_file = 'drytrpfi.100'

    site_list = generate_inputs()
    for site in site_list:
        out_dir_site = os.path.join(out_dir, site['name'])
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site)
        backcalc.back_calculate_management(site, input_dir, century_dir,
                                           out_dir_site, fix_file, n_years,
                                           vary, live_or_total, threshold,
                                           max_iterations)
                                  
if __name__ == "__main__":
    # generate_inputs()
    # calc_management()
    save_as = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Kenya/From_Jenny/Comparisons_with_CENTURY/back_calc_mgmt_9.13.16/calculated_management_summary.csv"
    # collect_results(save_as)
    summarize_calc_schedules(save_as)
