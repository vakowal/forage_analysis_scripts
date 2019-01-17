"""Test calculation of energy and protein requirements of pregnancy."""

import sys
import math

import pandas

sys.path.append(
    "C:/Users/ginge/Documents/Python/rangeland_production")

import forage_utils as forage
import forage_century_link_utils as cent
import freer_param as FreerParam


def test_diet_intermediates(herb_class):
    """Scraps of the function calc_diet_intermediates."""
    diet = forage.Diet()
    diet.If = 10.
    diet.DMDf = 0.64
    diet.CPIf = 1.

    # copied from calc_diet_intermediates()
    supp = forage.Supplement(
        FreerParam.FreerParamCattle('indicus'), 0, 0, 0, 0, 0, 0)
    site = forage.SiteInfo(1, 0)
    diet_interm = forage.DietIntermediates()
    MEIf = (17.0 * diet.DMDf - 2) * diet.If  # eq 31: herbage
    MEIs = (13.3 * supp.DMD + 23.4 * supp.EE + 1.32) * diet.Is  # eq 32
    FMEIs = (13.3 * supp.DMD + 1.32) * diet.Is  # eq 32, supp.EE = 0
    MEItotal = MEIf + MEIs  # assuming no intake of milk
    M_per_Dforage = MEIf / diet.If
    kl = herb_class.FParam.CK5 + herb_class.FParam.CK6 * M_per_Dforage  # eq 34
    km = (herb_class.FParam.CK1 + herb_class.FParam.CK2 * M_per_Dforage)  # eq 33 efficiency of energy use for maintenance
    Emove = herb_class.FParam.CM16 * herb_class.D * herb_class.W
    Egraze = herb_class.FParam.CM6 * herb_class.W * diet.If * \
             (herb_class.FParam.CM7 - diet.DMDf) + Emove
    Emetab = herb_class.FParam.CM2 * herb_class.W ** 0.75 * max(math.exp(
             -herb_class.FParam.CM3 * herb_class.A), herb_class.FParam.CM4)
    # eq 41, energy req for maintenance:
    MEm = (Emetab + Egraze) / km + herb_class.FParam.CM1 * MEItotal
    if herb_class.sex == 'castrate' or herb_class.sex == 'entire_m':
        MEm = MEm * 1.15
    if herb_class.sex == 'herd_average':
        MEm = MEm * 1.055
    if herb_class.sex == 'NA':
        MEm = (MEm + MEm * 1.15) / 2
    diet_interm.L = (MEItotal / MEm) - 1.

    # new calculations to test --- copy any corrections from here
    A_foet = 30.  # assumed days since conception
    RA = A_foet / herb_class.FParam.CP1
    BW = (
        1 - herb_class.FParam.CP4 + herb_class.FParam.CP4 *
        herb_class.Z) * herb_class.FParam.CP15 * herb_class.SRW
    BC_foet = 1  # assume condition of foetus is 1, ignoring weight
    # assume she is pregnant with one foetus
    MEc_num1 = (
        herb_class.FParam.CP8 * (herb_class.FParam.CP5 * BW) * BC_foet)
    MEc_num2 = (
        (herb_class.FParam.CP9 * herb_class.FParam.CP10) /
        herb_class.FParam.CP1)
    MEc_num3 = math.exp(
        herb_class.FParam.CP10 * (1 - RA) + herb_class.FParam.CP9 *
        (1 - math.exp(herb_class.FParam.CP10*(1 - RA))))
    MEc = (MEc_num1 * MEc_num2 * MEc_num3) / herb_class.FParam.CK8

    Pc_1 = herb_class.FParam.CP11 * (herb_class.FParam.CP5 * BW) * BC_foet
    Pc_2 = (
        (herb_class.FParam.CP12 * herb_class.FParam.CP13) /
        herb_class.FParam.CP1)
    Pc_3 = herb_class.FParam.CP13 * (1 - RA)
    Pc_4 = (
        herb_class.FParam.CP12 *
        (1 - math.exp(herb_class.FParam.CP13 * (1 - RA))))
    Pc_5 = math.exp(Pc_3 + Pc_4)
    Pc = Pc_1 * Pc_2 * Pc_5


if __name__ == "__main__":
    args = {
        'herbivore_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/cashmere_goats.csv"}

    herbivore_list = []
    herbivore_input = (pandas.read_csv(
        args[u'herbivore_csv'])).to_dict(orient='records')
    for h_class in herbivore_input:
        herd = forage.HerbivoreClass(h_class)
        herd.update()
        BC = 1
        herbivore_list.append(herd)

    test_diet_intermediates(herbivore_list[0])
