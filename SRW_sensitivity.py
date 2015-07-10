# quick and dirty sensitivity analysis: SRW and birth weight
# using other model inputs from Rubanza et al testing script

import os
import forage_utils as forage
import freer_param as FreerParam
import csv
import itertools
import numpy as np

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
site = forage.SiteInfo(total_SD, 1., -3.5)
prop_legume = 0.0
breed = 'Brahman'  # see documentation for allowable breeds; assumed to apply to all animal classes
sex = 'castrate'  # right now, we only deal with steers
A = 225.  # initial age (days)
W = 161.  # initial weight (Rubanza et al 2005)
herd_size = 1
DOY_start = 213
outdir = 'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\Sensitivity\SRW'
time_step = 'day'
force_supp = False
forage.set_time_step(time_step)
steps = 10

FParam = FreerParam.FreerParam(forage.get_general_breed(breed))

available_forage = forage.calc_feed_types(grass_list)
supp = forage.Supplement(FParam, 0.643, 0, 8.87, 0.031, 0.181, 0.75)
supp_available = 0

SRW_list = np.linspace(400, 700, steps)
Wbirth_list = np.linspace(27, 48, steps)
test_list = [SRW_list, Wbirth_list]

def one_step(FParam, DOY, herd, available_forage, prop_legume, supp_available,
    supp, intake = None):
    """One step of the forage model, if available forage does not change."""

    row = []
    max_intake = herd.calc_max_intake(FParam)

    if herd.Z < FParam.CR7:
        ZF = 1. + (FParam.CR7 - herd.Z)
    else:
        ZF = 1.

    if intake is not None:  # if forage intake should be forced
        diet = forage.Diet()
        diet.If = intake    
        diet.DMDf = available_forage[0].digestibility  # this is super hack-y
        diet.CPIf = intake * available_forage[0].crude_protein  # and only works with one type of available forage
        diet.Is = supp.DMO  # also force intake of all supplement offered
        diet_interm = forage.calc_diet_intermediates(FParam, diet, supp, herd, site,
            prop_legume, DOY)
    else:
        diet = forage.diet_selection_t2(ZF, prop_legume, supp_available, supp,
            max_intake, FParam, available_forage, force_supp)
        diet_interm = forage.calc_diet_intermediates(FParam, diet, supp, herd, site,
            prop_legume, DOY)
        reduced_max_intake = forage.check_max_intake(FParam, diet, diet_interm, herd,
            max_intake)
        if reduced_max_intake < max_intake:
            diet = forage.diet_selection_t2(ZF, prop_legume, supp_available, supp,
                reduced_max_intake, FParam, available_forage, force_supp)
            diet_interm = forage.calc_diet_intermediates(FParam, diet, supp, herd, site,
                prop_legume, DOY)

    delta_W = forage.calc_delta_weight(FParam, diet, diet_interm, supp, herd)
    #delta_W_step = forage.convert_daily_to_step(delta_W)
    herd.update(FParam, delta_W, forage.find_days_per_step())

    row.append(max_intake)    
    row.append(diet.If)
    row.append(diet.Is)
    row.append(diet.CPIf)
    row.append(diet_interm.MEItotal)
    row.append(delta_W)
    return row

out_name = os.path.join(outdir, 'summary.csv')
with open(out_name, 'wb') as out:
    writer = csv.writer(out, delimiter = ',')
    header = ['max_intake', 'intake_forage', 'intake_supp', 'CPI_forage',
             'ME_intake_total', 'daily_gain', 'SRW', 'Wbirth', 'step']
    writer.writerow(header)      
    
    for combination in itertools.product(*test_list):
        SRW = combination[0]
        Wbirth = combination[1]
        herd = forage.HerbivoreClass(FParam, breed, W, sex, A, 1., 'label',
            Wbirth, SRW)
        herd.update(FParam, 0, 0)
        
        # time of the experiment: 70 days
        for step in xrange(70):
           DOY = DOY_start + step
           row = one_step(FParam, DOY, herd, available_forage, prop_legume,
                          supp_available, supp)
           row.append(SRW)
           row.append(Wbirth)
           row.append(step)
           writer.writerow(row)                 






