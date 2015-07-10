# how many cattle would be supported by eating a certain amount?
# the amount of feed consumed was calculated by back-calculating management
# to match an empirical biomass target in CENTURY

import math
import os
import pandas
import forage_utils as forage
import forage_century_link_utils as cent

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
        herd_df = pandas.DataFrame(dict1)
        herd_sort = herd_df.sort(['event'])
        csvfile = os.path.join(outdir, site['name'] + 'herd_size_' + forage_quality + '.csv')
        herd_sort.to_csv(csvfile)