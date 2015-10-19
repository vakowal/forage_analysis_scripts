# quick and dirty sensitivity analysis: forage quality
# using other model inputs from Rubanza et al testing script

import os
import sys
if 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage' not in sys.path:
    sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage
import freer_param as FreerParam
import csv
import itertools
import numpy as np

total_SD = 1.
site = forage.SiteInfo(total_SD, 1., -3.5)
prop_legume = 0.0
breed = 'Brahman'  # see documentation for allowable breeds; assumed to apply to all animal classes
sex = 'castrate'  # right now, we only deal with steers
A = 225.  # initial age (days)
W = 161.  # initial weight (Rubanza et al 2005)
herd_size = 1
DOY_start = 60
outdir = 'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\Verification_calculations\Sensitivity\Forage_quality'
if not os.path.exists(outdir):
    os.makedirs(outdir)
time_step = 'day'
force_supp = False
forage.set_time_step(time_step)
steps = 100

FParam = FreerParam.FreerParam(forage.get_general_breed(breed))
supp = forage.Supplement(FParam, 0.643, 0, 8.87, 0.031, 0.181, 0.75)
supp_available = 0

DMD_list = np.linspace(0.1, 0.8, steps)
CP_list = np.linspace(0.01, 0.3, steps)
test_list = [DMD_list, CP_list]

out_name = os.path.join(outdir, 'summary.csv')
with open(out_name, 'wb') as out:
    writer = csv.writer(out, delimiter = ',')
    header = ['reduced_max_intake', 'max_intake', 'intake_forage', 
              'daily_gain', 'DMD', 'CP', 'step']
    writer.writerow(header)      
    
    for combination in itertools.product(*test_list):
        DMD = combination[0]
        CP = combination[1]
        grass1 = {
            'label': 'grass1',
            'type': 'C4',  # 'C3' or 'C4'
            'DMD_green': DMD,
            'DMD_dead': DMD,
            'cprotein_green': CP,
            'cprotein_dead': CP,
            'green_gm2': 2000.,
            'dead_gm2': 2000.,
            'percent_biomass': 1.,
            }
        grass_list = [grass1]
        available_forage = forage.calc_feed_types(grass_list)
        
        herb_class = forage.HerbivoreClass(FParam, breed, W, sex, A, total_SD)
        herb_class.update(FParam, 0, 0)
        row = forage.one_step(FParam, site, DOY_start, herb_class,
                             available_forage, prop_legume, supp_available,
                             supp)
        step = 0
        if len(row) == 2:
            row.insert(0, 'NA')
        row.append(DMD)
        row.append(CP)
        row.append(step)
        writer.writerow(row)                 






