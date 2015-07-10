# test livestock model with inputs from Panjaitan et al 2010

import math
import time
import os
import forage_utils as forage
import forage_century_link_utils as cent
import freer_param as FreerParam
import csv

grass1 = {
    'label': 'Speargrass',
    'type': 'C4',  # 'C3' or 'C4'
    'DMD_green': 0.465,
    'DMD_dead': 0.465,
    'cprotein_green': 0.0257,
    'cprotein_dead': 0.0257,
    'green_gm2': 2000.,
    'dead_gm2': 2000.,
    'percent_biomass': 1.,
    }
grass2 = {
    'label': 'Mitchell_grass',
    'type': 'C4',  # 'C3' or 'C4'
    'DMD_green': 0.407,
    'DMD_dead': 0.407,
    'cprotein_green': 0.0297,
    'cprotein_dead': 0.0297,
    'green_gm2': 2000.,
    'dead_gm2': 2000.,
    'percent_biomass': 1.,
    }
grass3 = {
    'label': 'Pangola_grass',
    'type': 'C4',  # 'C3' or 'C4'
    'DMD_green': 0.546,
    'DMD_dead': 0.546,
    'cprotein_green': 0.0755,
    'cprotein_dead': 0.0755,
    'green_gm2': 2000.,
    'dead_gm2': 2000.,
    'percent_biomass': 1.,
    }
grass4 = {
    'label': 'Ryegrass',
    'type': 'C4',  # 'C3' or 'C4'
    'DMD_green': 0.697,
    'DMD_dead': 0.697,
    'cprotein_green': 0.1998,
    'cprotein_dead': 0.1998,
    'green_gm2': 2000.,
    'dead_gm2': 2000.,
    'percent_biomass': 1.,
    }
grass_list = [grass1, grass2, grass3, grass4]

total_SD = 1.
site = forage.SiteInfo(total_SD, 1., -3.5)
prop_legume = 0.0
supp_available = 0
breed = 'Brahman'  # see documentation for allowable breeds; assumed to apply to all animal classes
sex = 'castrate'  # right now, we only deal with steers
A = 489.  # initial age (days)
W = 424.  # initial weight
herd_size = 1
DOY_start = 213
outdir = 'C:\Users\Ginger\Desktop\Rubanza_test'
time_step = 'day'
forage.set_time_step(time_step)
FParam = FreerParam.FreerParam(forage.get_general_breed(breed))
supp = forage.Supplement(FParam, 0.643, 0, 8.87, 0.031, 0.181, 0.75)

herd_list = []
forage_list = []
for index in xrange(4):
    grass = [grass_list[index]]
    available_forage = forage.calc_feed_types(grass)
    forage_list.append(available_forage)
    herd = forage.HerbivoreClass(FParam, breed, W, sex, A, total_SD)
    herd.update(FParam, 0, 0)
    herd_list.append(herd)

def one_step(FParam, DOY, herd, available_forage, prop_legume, supp_available,
    supp):
    """One step of the forage model, if available forage does not change."""

    row = []
    row.append(available_forage[0].label)
    max_intake = herd.calc_max_intake(FParam)

    if herd.Z < FParam.CR7:
        ZF = 1. + (FParam.CR7 - herd.Z)
    else:
        ZF = 1.

    diet = forage.diet_selection_t2(ZF, prop_legume, supp_available, supp,
        max_intake, FParam, available_forage)
    diet_interm = forage.calc_diet_intermediates(FParam, diet, supp, herd, site,
        prop_legume, DOY)
    reduced_max_intake = forage.check_max_intake(FParam, diet, diet_interm, herd,
        max_intake)
    if reduced_max_intake < max_intake:
        diet = forage.diet_selection_t2(ZF, prop_legume, supp_available, supp,
            reduced_max_intake, FParam, available_forage)
        diet_interm = forage.calc_diet_intermediates(FParam, diet, supp, herd, site,
            prop_legume, DOY)
    delta_W = forage.calc_delta_weight(FParam, diet, diet_interm, supp, herd)
    
    row.append(herd.W)
    row.append(max_intake)
    row.append(reduced_max_intake)
    row.append(diet.If)
    row.append(diet.CPIf)
    row.append(diet_interm.MEItotal)
    row.append(delta_W)
    
    #delta_W_step = forage.convert_daily_to_step(delta_W)
    herd.update(FParam, delta_W, forage.find_days_per_step())
    return row

out_name = os.path.join(outdir, 'summary.csv')
with open(out_name, 'wb') as out:
    writer = csv.writer(out, delimiter = ',')
    header = ['grass', 'weight_kg', 'max_intake', 'reduced_max_intake',
             'intake_forage', 'CPI_forage', 'ME_intake_total', 'daily_gain',
             'step']
    writer.writerow(header)              
    # time of the experiment: 7 days
    for step in xrange(7):
       DOY = DOY_start + step
    
       for i in xrange(len(grass_list)):
           available_forage = forage_list[i]
           herd = herd_list[i]
           row = one_step(FParam, DOY, herd, available_forage, prop_legume,
                 supp_available, supp)
           row.append(step)
           writer.writerow(row)                 






