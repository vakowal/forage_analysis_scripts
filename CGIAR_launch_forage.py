import os
import shutil
import pandas
import sys
sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def calibrate():
    """Calibrate the model to estimated sustainable capacity of the study area,
    according to the Yauyos Cochas report."""
    
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/calibrate_4.27.16/custom_calibration_2"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1991,
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
    for anim_type in ['cow', 'sheep', 'camelid']:
        grass_csv = os.path.join(input_dir, 'Pajonal_5_calib.csv')
        herbivore_csv = os.path.join(input_dir, anim_type + '_calib.csv')
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = herbivore_csv
        out_dir = os.path.join(outer_dir, '%s' % anim_type)
        forage_args['outdir'] = out_dir
        # if not os.path.exists(out_dir):
        forage.execute(forage_args)

def launch_baseline():
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/raw_5.2.16"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 204,
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
        'restart_yearly': 1,
    }
    for subbasin in range(1, 15):
        grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                 subbasin)
        herbivore_csv = os.path.join(input_dir, 'zero_sd.csv')
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = None
        out_dir = os.path.join(outer_dir, 's%d_zero_sd' % subbasin)
        forage_args['outdir'] = out_dir
        #if not os.path.exists(out_dir):
        forage.execute(forage_args)
                    
def launch_runs():
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Forage_model_results/test_5.9.16"
    forage_args = {
        'latitude': -12.55,
        'prop_legume': 0.0,
        'steepness': 1.,
        'DOY': 1,
        'start_year': 1993,
        'start_month': 1,
        'num_months': 204,
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
        'restart_yearly': 1,
    }
    for subbasin in range(1, 16):
        for anim_type in ['cow', 'sheep', 'camelid']:
            for sd_level in ['low', 'med', 'high']:
                forage_args['mgmt_threshold'] = 0.1
                out_dir = os.path.join(outer_dir, 's%d_%s_%s' %
                                       (subbasin, anim_type, sd_level))
                # summary_csv = os.path.join(out_dir, "summary_results.csv")
                # if os.path.isfile(summary_csv):
                    # df = pandas.read_csv(summary_csv)
                    # if df.shape[0] < 204:
                        # shutil.rmtree(out_dir)
                        # forage_args['mgmt_threshold'] = 0.5
                grass_csv = os.path.join(input_dir, 'Pajonal_%d.csv' %
                                         subbasin)
                herbivore_csv = os.path.join(input_dir, anim_type + '_' +
                                             sd_level + '_sd.csv')
                forage_args['grass_csv'] = grass_csv
                forage_args['herbivore_csv'] = herbivore_csv
                forage_args['outdir'] = out_dir
                if not os.path.exists(out_dir):
                    forage.execute(forage_args)
                        
if __name__ == "__main__":
    launch_runs()