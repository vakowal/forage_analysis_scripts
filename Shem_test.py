# Test forage model with inputs from Shem et al 1995

import math
import time
import os
import sys
sys.path.append(r'C:\Users\Ginger\Documents\Python\rangeland_production')
import forage_utils as forage
import forage_century_link_utils as cent
import freer_param as FreerParam
import csv
import pandas

def run_test():
    total_SD = 1.
    site = forage.SiteInfo(1., -3.25)
    prop_legume = 0.0
    supp_available = 0
    breed = 'Ayrshire'
    sex = 'entire_m'
    herd_size = 1
    DOY_start = 1
    outdir = r'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/Verification_calculations/Shem_et_al_1995/revisions_10_12'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    time_step = 'day'
    forage.set_time_step(time_step)
    supp_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Shem_et_al_1995_supp.csv"
    herbivore_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herbivore_Shem_et_al.csv"
    herbivore_input = (pandas.read_csv(herbivore_csv)).to_dict(orient='records')
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grasses_Shem_et_al_1995.csv"
    grass_list = (pandas.read_csv(grass_csv)).to_dict(orient='records')
    out_name = os.path.join(outdir, "summary.csv")

    supp_list = (pandas.read_csv(supp_csv)).to_dict(orient='records')
    supp_info = supp_list[0]
    supp = forage.Supplement(FreerParam.FreerParamCattle('indicus_x_taurus'),
                             supp_info['digestibility'],
                             supp_info['kg_per_day'], supp_info['M_per_d'],
                             supp_info['ether_extract'],
                             supp_info['crude_protein'],
                             supp_info['rumen_degradability'])
    if supp.DMO > 0.:
        supp_available = 1
    # supp_available = 0  # change to allow supplementation
    with open(out_name, 'wb') as out:
        writer = csv.writer(out, delimiter=',')
        header = ['red_max_intake', 'max_intake', 'intake_forage',
                  'intake_supp', 'daily_gain', 'grass_label',
                  'step', 'study']
        writer.writerow(header)
        for grass in grass_list:
            one_grass = [grass]
            available_forage = forage.calc_feed_types(one_grass)
            herbivore_list = []
            for h_class in herbivore_input:
                herd = forage.HerbivoreClass(h_class)
                herd.update()
                herbivore_list.append(herd)
                print "beginning weight: " + str(herd.W)
                print "beginning BC: " + str(herd.BC)
                print "SRW: " + str(herd.SRW)
            for step in xrange(60):
                DOY = DOY_start + step
                row = forage.one_step(site, DOY, herd,
                                      available_forage, prop_legume,
                                      supp_available, supp)
                row.append(grass['label'])
                row.append(step)
                row.append('Schem_et_al')
                writer.writerow(row)

def modify_herb_csv(herbivore_csv, CM2, CM12, CK13, CG2):
    """Modify calibration parameters of the herbivore csv used as input to the
    forage model."""
    
    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'CM2', CM2)
    df.set_value(0, 'CM12', CM12)
    df.set_value(0, 'CK13', CK13)
    df.set_value(0, 'CG2', CG2)
    df.to_csv(herbivore_csv)
    
def process_summary(pub_file, out_name):
    pub_dat = pandas.read_csv(pub_file)
    df = pandas.read_csv(out_name)
    by_grass = df.groupby('grass_label')
    mean_intake = by_grass['intake_forage'].mean()
    
    
if __name__ == "__main__":
    run_test()