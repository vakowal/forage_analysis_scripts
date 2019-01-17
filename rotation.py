# implement rotation with rangeland production model

import os
import sys
import math
import re
import pandas as pd
sys.path.append(
 'C:/Users/ginge/Documents/Python/rangeland_production')
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

    animal_dict = {'step': [], 'month': [], 'year': [],
                   'animal_diet_sufficiency': [],
                   'animal_intake_forage_per_indiv_kg': [],
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
        energy_intake = sub_df['{}_MEItotal'.format(anim_l)]
        energy_req = sub_df['{}_E_req'.format(anim_l)]
        diet_sufficiency = (energy_intake - energy_req) / energy_req
        animal_dict['animal_diet_sufficiency'].extend(diet_sufficiency)
        animal_dict['animal_intake_forage_per_indiv_kg'].extend(
                       sub_df['{}_intake_forage_per_indiv_kg'.format(anim_l)])

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

def continuous(forage_args, total_area_ha, num_animals, outdir,
               remove_months=None):
    """Continuous grazing, no rotation. If remove_months is supplied, grazing
    does not happen during those months. Remove_months should be a list of
    calendar months of the year (i.e. 1=January, 12=December)."""

    stocking_dens = float(num_animals) / total_area_ha
    modify_stocking_density(forage_args['herbivore_csv'], stocking_dens)
    forage_args['outdir'] = outdir

    if remove_months:
        # no grazing in remove_months
        assert forage_args['start_month'] == 1, "I can't figure out how to handle starting month other than 1"
        grz_months = range(0, forage_args['num_months'])
        for r in remove_months:
            grz_months = [m for m in grz_months if (m % 12) != (r - 1)]
    else:
        grz_months = range(forage_args['num_months'])
    forage_args['grz_months'] = grz_months
    forage.execute(forage_args)

def blind_rotation(forage_args, total_area_ha, n_pastures, num_animals,
                   outer_outdir, remove_months=None):
    """first stab at implementing rotation with the rangeland production
    model.  This version is "blind" because it is not responsive to pasture
    biomass. forage_args should contain arguments to run the model, and we
    assume that all pastures are run identically. n_pastures is how many
    pastures to rotate the herd among. We assume all pastures are equal size,
    and there is just one animal herd."""

    time_step = 'month'
    rot_length = 1  # if time triggers rotation, how many steps each pasture should be grazed at a time

    # calculate overall density, assuming equal pasture size and 1 herd
    pasture_size_ha = float(total_area_ha) / n_pastures
    stocking_dens = float(num_animals) / pasture_size_ha
    modify_stocking_density(forage_args['herbivore_csv'], stocking_dens)

    # generate grz_months for each pasture
    total_steps = forage_args['num_months']  # TODO daily time step ?
    grz_mo_list = generate_grz_months(total_steps, rot_length, n_pastures)

    # launch simulations
    for pidx in range(n_pastures):
        forage_args['outdir'] = os.path.join(outer_outdir, 'p_{}'.format(pidx))
        grz_months = grz_mo_list[pidx]
        if remove_months:
            # no grazing in remove_months
            assert forage_args['start_month'] == 1, "I can't figure out how to handle starting month other than 1"
            for r in remove_months:
                grz_months = [m for m in grz_months if (m % 12) != (r - 1)]
        forage_args['grz_months'] = grz_months
        if not os.path.exists(
                os.path.join(
                    outer_outdir, 'p_{}'.format(pidx), 'summary_results.csv')):
            forage.execute(forage_args)

    # collect results
    collect_rotation_results(forage_args, n_pastures, outer_outdir)

def get_max_biomass_pasture(outer_outdir, forage_args, n_pastures, date):
    """Identify the pasture with highest total biomass."""

    biom_list = []
    for pidx in range(n_pastures):
        outdir = os.path.join(outer_outdir, 'p_{}'.format(pidx))
        # find latest folder of outputs
        folder_list = [f for f in os.listdir(outdir) if
                        os.path.isdir(os.path.join(outdir, f))]
        folder_list.remove('CENTURY_outputs_spin_up')
        max_year = max([f[-4:] for f in folder_list])
        folder_list = [f for f in folder_list if f.endswith(max_year)]
        max_month = max([int(re.search(
                        'CENTURY_outputs_m(.+?)_y{}'.format(max_year),
                        f).group(1)) for f in folder_list])
        max_folder = os.path.join(outdir,
                                  'CENTURY_outputs_m{}_y{}'.format(
                                  max_month, max_year))
        output_files = [f for f in os.listdir(max_folder)
                        if f.endswith('.lis')]
        output_f = os.path.join(max_folder, output_files[0])
        outputs = cent.read_CENTURY_outputs(output_f,
                                            math.floor(date) - 1,
                                            math.ceil(date) + 1)
        outputs.drop_duplicates(inplace=True)
        total_biom = outputs.loc[date, 'aglivc'] + outputs.loc[date, 'stdedc']
        biom_list.append(total_biom)
    return biom_list.index(max(biom_list))

def rotation(forage_args, n_pastures, pasture_size_ha, num_animals,
             outer_outdir):
    """Rotation with allocation of rotated animals to pasture with highest
    total biomass."""

    time_step = 'month'
    rot_length = 1  # if time triggers rotation, how many steps each pasture should be grazed at a time

    # calculate overall density, assuming equal pasture size and 1 herd
    stocking_dens = float(num_animals) / pasture_size_ha
    modify_stocking_density(forage_args['herbivore_csv'], stocking_dens)

    # initialize grazing months list for each pasture
    grz_mo_list = [[] for n in range(n_pastures)]

    # launch simulations
    total_steps = forage_args['num_months']
    for rot_step in range(rot_length, total_steps, rot_length):  # one rotation length at a time
        forage_args['num_months'] = rot_step  # accumulate with each rot_step
        target_step = rot_step - rot_length  # target date to measure biomass; when grazing would start
        step_month = forage_args[u'start_month'] + target_step
        if step_month > 12:
            mod = step_month % 12
            if mod == 0:
                month = 12
                year = (step_month / 12) + forage_args[u'start_year'] - 1
            else:
                month = mod
                year = (step_month / 12) + forage_args[u'start_year']
        else:
            month = step_month
            year = (target_step / 12) + forage_args[u'start_year']
        date = year + float('%.2f' % (month / 12.))
        if rot_step == rot_length:
            max_pidx = 0  # use first pasture by default in first step
        else:
            max_pidx = get_max_biomass_pasture(outer_outdir, forage_args,
                                               n_pastures, date)
        grz_mo_list[max_pidx].extend(range(rot_step - rot_length, rot_step))  # next rotation step
        for pidx in range(n_pastures):  # launch each pasture
            forage_args['outdir'] = os.path.join(outer_outdir, 'p_{}'.format(pidx))
            forage_args['grz_months'] = grz_mo_list[pidx]
            forage.execute(forage_args)

    # collect results
    collect_rotation_results(forage_args, n_pastures, outer_outdir)

def calc_productivity_metrics(cont_dir, rot_dir):
    """Summarize difference in pasture and animal productivity between a
    rotated and continuous schedule."""

    cont_sum_csv = os.path.join(cont_dir, "summary_results.csv")
    rot_a_csv = os.path.join(rot_dir, "animal_summary.csv")
    rot_p_csv = os.path.join(rot_dir, "pasture_summary.csv")

    cont_sum = pd.read_csv(cont_sum_csv)
    cont_sum['date'] = cont_sum['year'] + 1./12 * cont_sum['month']
    cont_green_col = [f for f in cont_sum.columns if f.endswith('green_kgha')]
    assert len(cont_green_col) == 1, "Assume only one column matches"
    cont_green_col = cont_green_col[0]
    cont_dead_col = [f for f in cont_sum.columns if f.endswith('dead_kgha')]
    assert len(cont_dead_col) == 1, "Assume only one column matches"
    cont_dead_col = cont_dead_col[0]
    cont_sum['pasture_kgha_continuous'] = cont_sum[cont_green_col] + \
                                          cont_sum[cont_dead_col]

    cont_anim_col = [f for f in cont_sum.columns if f.endswith('gain_kg')]
    assert len(cont_anim_col) == 1, "Assume only one column matches"
    cont_sum.rename(columns={cont_anim_col[0]: 'gain_continuous'},
                    inplace=True)
    cont_df = cont_sum[['date', 'pasture_kgha_continuous', 'gain_continuous']]

    rot_p_df = pd.read_csv(rot_p_csv)
    rot_p_df['date'] = rot_p_df['year'] + 1./12 * rot_p_df['month']
    rot_grass_col = [f for f in rot_p_df.columns if f.endswith('total_kgha')]
    assert len(rot_grass_col) == 1, "Assume only one column matches"
    rot_p_df.rename(columns={rot_grass_col[0]: 'pasture_kgha_rot'},
                    inplace=True)
    rot_mean = rot_p_df.groupby('date')['pasture_kgha_rot'].mean().reset_index()
    summary_df = pd.merge(rot_mean, cont_df, how='inner', on='date')

    rot_a_df = pd.read_csv(rot_a_csv)
    rot_a_df['date'] = rot_a_df['year'] + 1./12 * rot_a_df['month']
    rot_a_df.rename(columns={'animal_gain': 'gain_rot'}, inplace=True)
    rot_a_df = rot_a_df[['date', 'gain_rot']]

    summary_df = pd.merge(summary_df, rot_a_df, how='inner', on='date')
    gain_diff = ((summary_df['gain_rot'].mean() -
                 summary_df['gain_continuous'].mean()) /
                 abs(summary_df['gain_continuous'].mean()))
    pasture_diff = ((summary_df['pasture_kgha_rot'].mean() -
                    summary_df['pasture_kgha_continuous'].mean()) /
                    (summary_df['pasture_kgha_continuous'].mean()))
    return gain_diff, pasture_diff
