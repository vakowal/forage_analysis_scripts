# Test the forage model: what is the stocking density that causes forage
# resources to crash?

import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.append('C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage

def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""
    
    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)

def check_min_biomass(summary_csv, label):
    """Find the minimum total biomass standing during the course of a forage
    model simulation."""
    
    df = pd.read_csv(summary_csv)
    live_biomass = label + "_green_kgha"
    dead_biomass = label + "_dead_kgha"
    df['total_biomass'] = df[live_biomass] + df[dead_biomass]
    min_biomass = min(df.total_biomass)
    return min_biomass

def get_label(csv):
    """Retrieve the label of a model input."""
    
    df = pd.read_csv(csv)
    assert len(df) == 1, "We can only handle one input type"
    label = df.iloc[0].label
    return label

def run_test(args, input_dir, outer_dir, herbivore_csv, grass_label):
    """Run the test to increase stocking density until biomass reaches the
    threshold, or for 10 iterations."""
    
    orig_herbivore_csv = os.path.join(os.path.dirname(herbivore_csv),
                                      os.path.basename(herbivore_csv)[:-4] +
                                      '_orig.csv')
    shutil.copyfile(herbivore_csv, orig_herbivore_csv)
    args['herbivore_csv'] = herbivore_csv
    sd_list =  np.linspace(0.17535, 2, num=6)
    try:
        for sd in sd_list:
            modify_stocking_density(herbivore_csv, sd)
            args['outdir'] = os.path.join(outer_dir, "outputs_%f" % sd)
            if not os.path.exists(args['outdir']):
                os.makedirs(args['outdir'])
            shutil.copyfile(herbivore_csv, os.path.join(args['outdir'],
                                                        'herbivore.csv'))
            forage.execute(args)
    finally:
        shutil.copyfile(orig_herbivore_csv, herbivore_csv)
        os.remove(orig_herbivore_csv)

def collect_results(outer_dir, herd_label, grass_label):
    """Collect results and format in a single data frame for analysis and
    plotting."""
    
    summary_dict = {'stocking_density': [],
                    'date': [],
                    'animal_weight_kg': [],
                    'animal_gain_kg': [],
                    'above_total_biomass_kg_ha': [],
                    'total_offtake_kg': [],
                    'precip_cm': [],
                    'h2o_avail_cm': [],
                    'h2o_deep_storage_cm': [],
                    'stream_flow_cm': [],
                    'below_biomass_gm2': [],
    }
    for folder_name in os.listdir(outer_dir):
        folder = os.path.join(outer_dir, folder_name)
        sd_df = pd.read_csv(os.path.join(folder, 'herbivore.csv'))
        sd = sd_df.iloc[0].stocking_density
        for idx in xrange(12):
            summary_dict['stocking_density'].append(sd)
        cent_out = os.path.join(folder, "CENTURY_outputs_m12_y2015",
                                "%s.lis" % grass_label)
        cent_df = pd.read_fwf(cent_out)
        cent_df = cent_df.loc[cent_df['time'] >= 2015]
        cent_df = cent_df.loc[cent_df['time'] < 2016]
        # we assume nlaypg (num soil layers for plant growth) == 4
        cent_df['h2o_pg_cm'] = cent_df['asmos(1)'] + cent_df['asmos(2)'] +\
                               cent_df['asmos(3)'] + cent_df['asmos(4)']
        for item in cent_df['time'].tolist():
            summary_dict['date'].append(item)
        for item in cent_df['rain'].tolist():
            summary_dict['precip_cm'].append(item)
        for item in cent_df['h2o_pg_cm'].tolist():
            summary_dict['h2o_avail_cm'].append(item)
        for item in cent_df['asmos(5)'].tolist():
            summary_dict['h2o_deep_storage_cm'].append(item)
        for item in cent_df['stream(1)'].tolist():
            summary_dict['stream_flow_cm'].append(item)
        for item in cent_df['bglivc'].tolist():
            summary_dict['below_biomass_gm2'].append(item * 2.5)
        
        results_df = pd.read_csv(os.path.join(folder, 'summary_results.csv'))
        results_df['total_biomass'] = results_df['%s_green_kgha' %
                                      grass_label] + results_df[
                                      '%s_dead_kgha' % grass_label]
        for item in results_df['%s_kg' % herd_label].tolist():
            summary_dict['animal_weight_kg'].append(item)
        for item in results_df['%s_gain_kg' % herd_label].tolist():
            summary_dict['animal_gain_kg'].append(item)
        for item in results_df['total_offtake'].tolist():
            summary_dict['total_offtake_kg'].append(item)
        for item in results_df['total_biomass'].tolist():
            summary_dict['above_total_biomass_kg_ha'].append(item)
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.to_csv(os.path.join(outer_dir, 'summary.csv'))
        
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
        'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
        'outdir': "C://Users//Ginger//Documents//Python//Output",
        'template_level': 'GL',
        'fix_file': 'drytrpfi.100',
        'user_define_protein': 1,
        'user_define_digestibility': 1,
        'herbivore_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herbivores.csv",
        'grass_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/grass_.64_.1_d_precip.csv",
        'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
        'input_dir': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input",
    }
    grass_label = get_label(args['grass_csv'])
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs"
    outer_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Output/Stocking_density_test/sustainable_limit_test/doubled_precip"
    herbivore_csv = os.path.join(input_dir, "herd_average.csv")
    herd_label = get_label(herbivore_csv)
    
    run_test(args, input_dir, outer_dir, herbivore_csv, grass_label)
    collect_results(outer_dir, herd_label, grass_label)