import os
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

if __name__ == "__main__":
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 2015,
        'start_month': 1,
        'num_months': 12,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': "",
        'template_level': 'GL',
        'fix_file': 'gfix.100',
        'user_define_protein': 0,
        'user_define_digestibility': 0,
        'herbivore_csv': "",
        'grass_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs/Pajonal.csv",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
    }
    # for climate zone in 0, 1, 2
    # for soil zone in 0, 1, 2
    # for animal type in 0, 1, 2
    for sd_level in ['low', 'med', 'high']:
        herbivore_csv = os.path.join(input_dir, 'cattle_' + sd_level +
                                     '_sd.csv')
        out_dir = os.path.join(outer_dir, 'cattle_' + sd_level)
        forage_args['herbivore_csv'] = herbivore_csv
        forage_args['outdir'] = out_dir
        forage.execute(forage_args)