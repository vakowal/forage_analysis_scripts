# Test the forage model with livestock herds of different herd composition

import os
import sys

sys.path.append('C:/Users/ginge/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

if __name__ == "__main__":
    args = {
    'latitude': 0.083,
    'prop_legume': 0.0,
    'breed': 'Brahman',  # see documentation for allowable breeds; assumed to apply to all animal classes
    'steepness': 1.,
    'DOY': 1,
    'start_year': 2015,
    'start_month': 1,
    'num_months': 12,
    'mgmt_threshold': 0.1,
    'century_dir': 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
    'outdir': "C://Users//Ginger//Documents//Python//Output",
    'template_level': 'GL',
    'fix_file': 'drytrpfi.100',
    'user_define_protein': 1,
    'user_define_digestibility': 1,
    'herbivore_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herbivores.csv",
    'grass_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass_.64_.1.csv",
    'supp_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
    'input_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input",
    }

    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs"
    outer_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Output/Stocking_density_test/Animal_type_test"
    for idx in [1, 2, 3, 4, 5]:
        args['herbivore_csv'] = os.path.join(input_dir,
                                             "herbivores_%d.csv" % idx)
        args['outdir'] = os.path.join(outer_dir, "outputs_%d" % idx)
        if not os.path.exists(args['outdir']):
            os.makedirs(args['outdir'])
        forage.execute(args)