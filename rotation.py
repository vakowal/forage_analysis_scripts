# implement rotation with rangeland production model

import os
import sys
import pandas as pd
sys.path.append(
 'C:/Users/Ginger/Documents/Python/rangeland_production')
import forage_century_link_utils as cent
import forage


def generate_grz_months(total_steps, rot_length, n_pastures):
    """Generate a list of steps when grazing should take place for each
    pasture, assuming the trigger to rotate is time, and assuming equal time
    spent at each pasture."""
    
    grz_mo_list = []
    ret_interval = rot_length * n_pastures
    for pasture in range(n_pastures):
        nested_months = [range(r, r + rot_length) for r in
                         range(pasture*rot_length, total_steps, ret_interval)]
        grz_months = [item for sublist in nested_months for item in sublist]
        grz_months = [m for m in grz_months if m < total_steps]
        grz_mo_list.append(grz_months)
    return grz_mo_list

def collect_rotation_results(forage_args, n_pastures, outer_outdir):
    """Summarize results across separate simulations representing multiple
    pastures used in rotation.  Generate two tables: One for the animal herd,
    giving weight gain in each step irregardless of pasture, and one for
    plant biomass, giving biomass in each step in each pasture."""
    
    animal_dict = {'step': [], 'month': [], 'year': [], 'animal_gain': [],
                   'animal_intake_forage_per_indiv_kg': [], 'animal_kg': [],
                   'total_offtake': [], 'pasture_index': []}
    plant_dict = {'step': [], 'month': [], 'year': [], 'pasture_index': [],
                  'total_offtake': []}
    grass_labels = pd.read_csv(forage_args['grass_csv']).label
    anim_l = pd.read_csv(forage_args['herbivore_csv']).iloc[0].label
    for g_label in grass_labels:
        plant_dict['{}_dead_kgha'.format(g_label)] = []
        plant_dict['{}_green_kgha'.format(g_label)] = []
        plant_dict['{}_total_kgha'.format(g_label)] = []
        
    for pidx in range(n_pastures):
        sum_df = pd.read_csv(os.path.join(outer_outdir,
                                              'p_{}'.format(pidx),
                                              'summary_results.csv'))
        sub_df = sum_df.loc[(sum_df["total_offtake"] > 0)]
        animal_dict['step'].extend(sub_df.step)
        animal_dict['month'].extend(sub_df.month)
        animal_dict['year'].extend(sub_df.year)
        animal_dict['total_offtake'].extend(sub_df.total_offtake)
        animal_dict['pasture_index'].extend([pidx] * len(sub_df.step))
        animal_dict['animal_gain'].extend(sub_df['{}_gain_kg'.format(anim_l)])
        animal_dict['animal_intake_forage_per_indiv_kg'].extend(
                       sub_df['{}_intake_forage_per_indiv_kg'.format(anim_l)])
        animal_dict['animal_kg'].extend(sub_df['{}_kg'.format(anim_l)])
        
        plant_dict['step'].extend(sum_df.step)
        plant_dict['month'].extend(sum_df.month)
        plant_dict['year'].extend(sum_df.year)
        plant_dict['total_offtake'].extend(sum_df.total_offtake)
        plant_dict['pasture_index'].extend([pidx] * len(sum_df.step))
        for g_label in grass_labels:
            plant_dict['{}_dead_kgha'.format(g_label)].extend(
                                       sum_df['{}_dead_kgha'.format(g_label)])
            plant_dict['{}_green_kgha'.format(g_label)].extend(
                                      sum_df['{}_green_kgha'.format(g_label)])
            total_kgha = sum_df['{}_green_kgha'.format(g_label)] + \
                                       sum_df['{}_dead_kgha'.format(g_label)]
            plant_dict['{}_total_kgha'.format(g_label)].extend(total_kgha)
    import pdb; pdb.set_trace()
    p_df = pd.DataFrame(plant_dict)
    p_df.to_csv(os.path.join(outer_outdir, 'pasture_summary.csv'), index=False)
    a_df = pd.DataFrame(animal_dict)
    a_df.to_csv(os.path.join(outer_outdir, 'animal_summary.csv'), index=False)

def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""
    
    df = pd.read_csv(herbivore_csv)
    try:
        df = df.set_index(['index'])
    except:
        df['index'] = 0
        df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)
    
def blind_rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
                    outer_outdir):
    """first stab at implementing rotation with the rangeland production
    model.  This version is "blind" because it is not responsive to pasture
    biomass. forage_args should contain arguments to run the model, and we
    assume that all pastures are run identically. n_pastures is how many 
    pastures to rotate the herd among. We assume all pastures are equal size,
    and there is just one animal herd."""
    
    time_step = 'month'
    rot_length = 1  # if time triggers rotation, how many steps each pasture should be grazed at a time
    
    # calculate overall density, assuming equal pasture size and 1 herd
    stocking_dens = float(num_animals) / pasture_size_ha
    modify_stocking_density(forage_args['herbivore_csv'], stocking_dens)
    
    # generate grz_months for each pasture
    total_steps = forage_args['num_months']  # TODO daily time step ?
    grz_mo_list = generate_grz_months(total_steps, rot_length, n_pastures)
    
    # launch simulations
    for pidx in range(n_pastures):
        forage_args['outdir'] = os.path.join(outer_outdir, 'p_{}'.format(pidx))
        forage_args['grz_months'] = grz_mo_list[pidx]
        forage.execute(forage_args)
    
    # collect results
    collect_rotation_results(forage_args, n_pastures, outer_outdir)

