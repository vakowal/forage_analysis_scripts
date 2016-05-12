## explore potential for different animal types to facilitate each other
## through differential diet selection

import os
import shutil
import pandas
import math
from operator import attrgetter
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_utils as forage
import freer_param as FreerParam

def write_grass_csv(csv, abun_mean, cp_mean, abun_ratio, cp_ratio):
    """Write a csv describing two grass types that relate to each other as
    described by the abundance ratio and the crude protein ratio."""
    
    df = pandas.read_csv(csv)
    

def diet_selection_t2(ZF, HR, prop_legume, supp_available, supp, Imax, FParam,
                      available_forage, idx, f_w=0, q_w=0, force_supp=None):
    """Perform diet selection for an individual herbivore, tier 2.  This
    function calculates relative availability, F (including factors like
    pasture height and animal mouth size) and relative ingestibility, RQ 
    including factors like pasture digestibility and proportion legume in the
    sward) to select a preferred mixture of forage types, including supplement
    if offered.  Available forage must be supplied to the function in an
    ordered list such that available_forage[0] is of highest digestibility.

    Returns daily intake of forage (kg; including seeds), daily intake of
    supplement (kg), average dry matter digestibility of forage, and average
    crude protein intake from forage."""

    available_forage = sorted(available_forage, reverse=True,
                              key=attrgetter('digestibility'))
    diet_selected = forage.Diet()
    if Imax == 0:
        for f_index in range(len(available_forage)):
            f_label = available_forage[f_index].label + ';' +\
                        available_forage[f_index].green_or_dead
            diet_selected.intake[f_label] = 0.
        return diet_selected

    F = list()
    RR = list()
    RT = list()
    HF = list()
    RQ = list()
    R = list()
    R_w = list()
    I = list()
    sum_prev_classes = 0.
    UC = 1.
    supp_selected = 0

    for f_index in range(len(available_forage)):
        RQ.append(1. - FParam.CR3 * (FParam.CR1 - (1. - prop_legume)  # eq 21
                  * available_forage[f_index].SF -
                  available_forage[f_index].digestibility))
        if supp_available:
            if RQ[f_index] <= supp.RQ or force_supp:
                supp_selected = 1
                supp_available = 0
                Fs = min((supp.DMO / Imax) / supp.RQ, UC, FParam.CR11 /
                          supp.M_per_D)  # eq 23
                sum_prev_classes += Fs
                UC = max(0., 1. - sum_prev_classes)
        HF.append(1. - FParam.CR12 + FParam.CR12 * HR[f_index])  # eq 18
        RT.append(1. + FParam.CR5 * math.exp(-(1. + FParam.CR13 *  # eq 17
            available_forage[f_index].rel_availability) *
            (FParam.CR6 * HF[f_index] * ZF * available_forage[f_index].biomass)
            ** 2))
        RR.append(1. - math.exp(-(1. + FParam.CR13 *  # eq 16
            available_forage[f_index].rel_availability) * FParam.CR4 *
            HF[f_index] * ZF * available_forage[f_index].biomass))
        F.append(UC * RR[f_index] * RT[f_index])  # eq 14
        sum_prev_classes += F[f_index]
        UC = max(0., 1. - sum_prev_classes)  # eq 15
    # import pdb; pdb.set_trace()
    for f_index in range(len(available_forage)):
        # original GRAZPLAN formulation
        R.append(F[f_index] * RQ[f_index] * (1. + FParam.CR2 * sum_prev_classes
                ** 2 * prop_legume))  # eq 20
        # weight proportional intake by quantity weight and quality weight
        R_w.append(R[f_index] + F[f_index]*f_w + RQ[f_index]*q_w)
    # rescale weighted proportions to retain original sum of R
    for f_index in range(len(available_forage)):
        R_w[f_index] = (R_w[f_index] / sum(R_w)) * sum(R)
    for f_index in range(len(available_forage)):    
        I.append(Imax * R_w[f_index])  # eq 27
        diet_selected.DMDf += (I[f_index] *
                               available_forage[f_index].digestibility)
        diet_selected.CPIf += (I[f_index] *
                               available_forage[f_index].crude_protein)
        diet_selected.If += I[f_index]

        # stash the amount consumed of each forage type
        f_label = available_forage[f_index].label + ';' +\
                  available_forage[f_index].green_or_dead
        diet_selected.intake[f_label] = I[f_index]
    print "diet %d: " % idx
    print "quality weight: %f, availability weight: %f" % (q_w, f_w)
    print "F: "
    print F
    print "RQ: "
    print RQ
    print "R: "
    print R
    print "R weighted: "
    print R_w
    diet_selected.DMDf = diet_selected.DMDf / diet_selected.If
    if supp_selected:
        Rs = Fs * supp.RQ  # eq 25
        diet_selected.Is = Imax * Rs  # eq 30
    return diet_selected

def different_diets():
    """Create different diets for two hypothetical animal types."""
    time_step = 'month'
    forage.set_time_step(time_step)
    total_SD = 2
    prop_legume = 0
    DOY = 100
    supp_available = 0
    site = forage.SiteInfo(1., -3.25)
    
    supp_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Shem_et_al_1995_supp.csv"
    supp_list = (pandas.read_csv(supp_csv)).to_dict(orient='records')
    supp_info = supp_list[0]
    supp = forage.Supplement(FreerParam.FreerParamCattle('indicus'),
                             supp_info['digestibility'],
                             supp_info['kg_per_day'], supp_info['M_per_d'],
                             supp_info['ether_extract'],
                             supp_info['crude_protein'],
                             supp_info['rumen_degradability'])
                             
    grass_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grasses_diet_illustration.csv"
    grass_list = (pandas.read_csv(grass_csv)).to_dict(orient='records')
    available_forage = forage.calc_feed_types(grass_list)
    
    herbivore_csv = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herbs_diet_illustration.csv"
    herbivore_input = (pandas.read_csv(herbivore_csv)).to_dict(orient='records')
    herbivore_list = []
    for h_class in herbivore_input:
        herd = forage.HerbivoreClass(h_class)
        herd.update()
        herbivore_list.append(herd)
    
    avail_weights = [0.1, 0.9]
    qual_weights = [1.0 - w for w in avail_weights]
    
    diet_dict = {}        
    for idx in xrange(len(herbivore_list)):
        herb_class = herbivore_list[idx]
        f_w = avail_weights[idx]
        rq_w = qual_weights[idx]
        herb_class.calc_distance_walked(total_SD, site.S, available_forage)
        max_intake = herb_class.calc_max_intake()

        ZF = herb_class.calc_ZF()
        HR = forage.calc_relative_height(available_forage)
        diet = diet_selection_t2(ZF, HR, prop_legume,
                                        supp_available, supp,
                                        max_intake, herb_class.FParam,
                                        available_forage, idx, f_w, rq_w)
        diet_interm = forage.calc_diet_intermediates(
                        diet, supp, herb_class, site,
                        prop_legume, DOY)
        # if herb_class.type != 'hindgut_fermenter':
            # reduced_max_intake = forage.check_max_intake(diet,
                                                         # diet_interm,
                                                         # herb_class,
                                                         # max_intake)
            # if reduced_max_intake < max_intake:
                # diet = forage.diet_selection_t2(ZF, HR,
                                                # args[u'prop_legume'],
                                                # supp_available, supp,
                                                # reduced_max_intake,
                                                # herb_class.FParam,
                                                # available_forage)
        diet_dict[herb_class.label] = diet
    # forage.reduce_demand(diet_dict, stocking_density_dict,
                         # available_forage)
    diet_interm = forage.calc_diet_intermediates(
                                        diet, supp, herb_class, site,
                                        prop_legume, DOY)
    delta_W = forage.calc_delta_weight(diet_interm, herb_class)
    delta_W_step = forage.convert_daily_to_step(delta_W)
    herb_class.update(delta_weight=delta_W_step,
                      delta_time=forage.find_days_per_step())

if __name__ == "__main__":
    different_diets()