import os
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

if __name__ == "__main__":
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/raw_4.26.16"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_month': 1,
        'num_months': 12,
        'mgmt_threshold': 0.1,
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': "",
        'template_level': 'GL',
        'fix_file': 'gfix.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "",
        'grass_csv': "",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': input_dir,
    }
    for subbasin in range(1, 15):
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sd_level in ['low', 'med', 'high']:
                grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                         subbasin)
                herbivore_csv = os.path.join(input_dir, anim_type + '_' +
                                             sd_level + '_sd.csv')
                forage_args['grass_csv'] = grass_csv
                forage_args['herbivore_csv'] = herbivore_csv
                for year in range(1991, 2010):
                    forage_args['start_year'] = year
                    out_dir = os.path.join(outer_dir, 's%d_%s_%s_%s' %
                                           (subbasin, anim_type, sd_level, 
                                            year))
                    forage_args['outdir'] = out_dir
                    # if not os.path.exists(out_dir):
                    forage.execute(forage_args)