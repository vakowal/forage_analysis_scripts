# test livestock model with inputs from Rubanza et al 2005

import math
import time
import os
import sys
if 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage' not in sys.path:
    sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage
import forage_century_link_utils as cent
import freer_param as FreerParam
import csv

grass1 = {
    'label': 'grass1',
    'type': 'C4',  # 'C3' or 'C4'
    'DMD_green': 0.541,
    'DMD_dead': 0.541,
    'cprotein_green': 0.023,
    'cprotein_dead': 0.023,
    'green_gm2': 2000.,
    'dead_gm2': 2000.,
    'percent_biomass': 1.,
    }
grass_list = [grass1]
total_SD = 1.
site = forage.SiteInfo(1., -3.5)
prop_legume = 0.0
breed = 'Brahman'  # see documentation for allowable breeds; assumed to apply to all animal classes
sex = 'castrate'  # right now, we only deal with steers
A = 225.  # initial age (days)
W = 161.  # initial weight (Rubanza et al 2005)
herd_size = 1
Wbirth = 34.7
SRW = 297
DOY_start = 213
outdir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\Rubanza_et_al_2005'
time_step = 'day'
force_supp = False
forage.set_time_step(time_step)


available_forage = forage.calc_feed_types(grass_list)

# different levels of supplement (kg per day) offered to experimental groups
supp_kg_list = [0, 0.4, 0.6, 0.8]
initial_weight_list = [162., 151., 164., 164.]

# forage intake published by Rubanza
intake_list = [3.1, 3.45, 3.72, 3.82]
supp_list = []
herd_list = []
for supp_index in xrange(4):
    supp = forage.Supplement(FreerParam.FreerParamCattle('indicus'), 0.643,
                             supp_kg_list[supp_index], 8.87,
                             0.031, 0.181, 0.75)
    supp_list.append(supp)
    herd = forage.HerbivoreClass('B_indicus', initial_weight_list[supp_index],
                                 sex, A, 1, SRW, Wbirth=Wbirth)
    herd.update()
    herd_list.append(herd)

def one_step(site, DOY, herb_class, available_forage, prop_legume,
             supp_available, supp, intake=None, force_supp=None):
    """One step of the forage model, if available forage does not change."""

    row = []
    row.append(supp.DMO)
    herb_class.calc_distance_walked(herb_class.stocking_density, site.S,
                                    available_forage)
    max_intake = herb_class.calc_max_intake()

    ZF = herb_class.calc_ZF()
    HR = forage.calc_relative_height(available_forage)
    if intake is not None:  # if forage intake should be forced
        diet = forage.Diet()
        diet.If = intake
        diet.DMDf = available_forage[0].digestibility  # this is super hack-y
        diet.CPIf = intake * available_forage[0].crude_protein  # and only works with one type of available forage
        diet.Is = supp.DMO  # also force intake of all supplement offered
        diet_interm = forage.calc_diet_intermediates(diet, supp, herb_class,
                                                     site, prop_legume, DOY)
        row.append('NA')
    else:
        diet = forage.diet_selection_t2(ZF, HR, prop_legume, supp_available,
                                        supp, max_intake, herb_class.FParam,
                                        available_forage, force_supp)
        diet_interm = forage.calc_diet_intermediates(diet, supp, herb_class,
                                                     site, prop_legume, DOY)
        if herb_class.type != 'hindgut_fermenter':
            reduced_max_intake = forage.check_max_intake(diet, diet_interm,
                                                         herb_class,
                                                         max_intake)
            row.append(reduced_max_intake)
            if reduced_max_intake < max_intake:
                diet = forage.diet_selection_t2(
                                         ZF, HR, prop_legume, supp_available,
                                         supp, reduced_max_intake,
                                         herb_class.FParam, available_forage,
                                         force_supp)
                diet_interm = forage.calc_diet_intermediates(
                                         diet, supp, herb_class, site,
                                         prop_legume, DOY)
    delta_W = forage.calc_delta_weight(diet_interm, herb_class)
    delta_W_step = forage.convert_daily_to_step(delta_W)
    herb_class.update(delta_weight=delta_W_step,
                      delta_time=forage.find_days_per_step())

    row.append(max_intake)
    row.append(diet.If)
    row.append(diet.Is)
    row.append(diet.CPIf)
    row.append(diet_interm.MEItotal)
    row.append(delta_W)
    return row

out_name = os.path.join(outdir, 'summary_force_supp_force_intake.csv')
with open(out_name, 'wb') as out:
    writer = csv.writer(out, delimiter=',')
    header = ['supp_level', 'reduced_max_intake', 'max_intake',
              'intake_forage', 'intake_supp', 'CPI_forage', 'ME_intake_total',
              'daily_gain', 'step']
    writer.writerow(header)
    for i in xrange(len(supp_list)):
       supp = supp_list[i]
       if supp.DMO == 0:
           supp_available = 0
       else:
           supp_available = 1
       herd = herd_list[i]
       intake = intake_list[i]
       # time of the experiment: 70 days
       for step in xrange(70):
           DOY = DOY_start + step
           row = one_step(site, DOY, herd, available_forage, prop_legume,
                          supp_available, supp, intake=intake, force_supp=1)
           row.append(step)
           writer.writerow(row)









