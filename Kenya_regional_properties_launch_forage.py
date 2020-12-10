# run forage model on regional properties

import os
import sys
import re
import shutil
import math
import random
from tempfile import mkstemp
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append("C:/Users/ginge/Documents/Python/rangeland_production_11_26_18")
# import back_calculate_management as backcalc
# import forage_century_link_utils as cent
import forage


def edit_conception_month(herbivore_csv, conception_step):
    """Edit the conception month for breeding cows in the herbivore csv."""
    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    df.loc[
        df['conception_step'].notnull(), 'conception_step'] = conception_step
    df.to_csv(herbivore_csv)


def calc_stocking_density(herbivore_csv, overall_sd):
    """Calculate the stocking density of each age/sex class.

    Stocking density of each age/sex herbivore class is calculated by
    multiplying the overall stocking density (i.e., total number animals per
    ha) by the proportion of the herd that each age/sex class represents.

    Modifies:
        the herbivore csv file

    Returns:
        None
    """
    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.stocking_density = overall_sd * df.proportion_of_herd
    df.to_csv(herbivore_csv)


def modify_stocking_density(herbivore_csv, new_sd):
    """Modify the stocking density in the herbivore csv used as input to the
    forage model."""

    df = pd.read_csv(herbivore_csv)
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one herbivore type"
    df['stocking_density'] = df['stocking_density'].astype(float)
    df.set_value(0, 'stocking_density', new_sd)
    df.to_csv(herbivore_csv)

def default_forage_args():
    """Default args for the forage model for regional properties."""

    forage_args = {
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': 2014,
            'start_month': 1,
            'num_months': 24,
            'mgmt_threshold': 300,
            'century_dir': 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'supp_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
            }
    return forage_args

def id_failed_simulation(result_dir, num_months):
    """Test whether a simulation completed the specified num_months. Returns 0
    if the simulation failed, 1 if the simulation succeeded."""

    try:
        sum_csv = os.path.join(result_dir, 'summary_results.csv')
        sum_df = pd.read_csv(sum_csv)
    except:
        return 0
    if len(sum_df) == (num_months + 1):
        return 1
    else:
        return 0

def summarize_offtake():
    """Compare offtake among several sets of simulations differing in inputs."""

    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")

    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    cp_opts = ['varying', 'constant']
    marg_dict = {'site': [], 'avg_offtake_varying_cp': [],
                 'avg_offtake_constant_cp': []}
    for site in site_list:
        marg_dict['site'].append(site['name'])
        for cp_o in cp_opts:
            inner_folder_name = "herd_avg_uncalibrated_0.3_{}_cp_GL".format(cp_o)
            inner_dir = os.path.join(outer_dir, inner_folder_name)
            outdir = os.path.join(inner_dir,
                                  'site_{:d}'.format(int(site['name'])))
            sum_csv = os.path.join(outdir, 'summary_results.csv')
            sum_df = pd.read_csv(sum_csv)
            subset = sum_df.loc[sum_df['year'] > 2013]
            avg_offtake = subset.total_offtake.mean()
            if cp_o == 'varying':
                marg_dict['avg_offtake_varying_cp'].append(avg_offtake)
            else:
                marg_dict['avg_offtake_constant_cp'].append(avg_offtake)
    df = pd.DataFrame(marg_dict)
    summary_csv = os.path.join(outer_dir, 'offtake_summary.csv')
    df.to_csv(summary_csv)

def summarize_remaining_biomass():
    """Summarize monthly biomass remaining at each site, assuming allowable use
    of 50% (cows + other grazers can offtake 50%). Calculate how many gazelle
    equivalents would be supported by forage remaining after cows offtake."""

    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")

    outer_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    cp_opts = ['cp']  # ['varying', 'constant']
    marg_dict = {'site': [], 'month': [], 'year': [], 'remaining_biomass': [],
                 'gazelle_equivalents': [], 'cp_option': []}
    for site in site_list:
        for cp_o in cp_opts:
            # inner_folder_name = "herd_avg_uncalibrated_0.3_{}_cp_GL".format(cp_o)
            inner_folder_name = "herd_avg_uncalibrated_constant_cp_GL_est_densities"
            inner_dir = os.path.join(outer_dir, inner_folder_name)
            outdir_folder = [f for f in os.listdir(inner_dir) if
                             f.startswith('site_{:d}_'.format(int(site['name'])))]
            try:
                outdir = os.path.join(inner_dir, outdir_folder[0])
            except:
                continue
            # outdir = os.path.join(inner_dir,
                                  # 'site_{:d}'.format(int(site['name'])))
            sum_csv = os.path.join(outdir, 'summary_results.csv')
            sum_df = pd.read_csv(sum_csv)
            subset = sum_df.loc[sum_df['year'] > 2013]
            subset.total_biomass = subset['{:.0f}_green_kgha'.format(site['name'])] + \
                                   subset['{:.0f}_dead_kgha'.format(site['name'])]
            subset.available = subset.total_biomass / 2
            subset.remaining = subset.available - subset.total_offtake
            gazelle_equiv = subset.remaining / 56.29
            marg_dict['site'].extend([site['name']] * len(subset.month))
            marg_dict['month'].extend(subset.month.tolist())
            marg_dict['year'].extend(subset.year.tolist())
            marg_dict['remaining_biomass'].extend(subset.remaining.tolist())
            marg_dict['gazelle_equivalents'].extend(gazelle_equiv.tolist())
            marg_dict['cp_option'].extend([cp_o] * len(subset.month))
    import pdb; pdb.set_trace()
    df = pd.DataFrame(marg_dict)
    summary_csv = os.path.join(outer_dir, 'biomass_remaining_summary.csv')
    df.to_csv(summary_csv)


def fill_dict(d_fill, fill_val):
    """Fill a dictionary with fill_val so that it can be converted to a pandas
    data frame and written to csv."""

    max_len = 0
    for key in d_fill.keys():
        if len(d_fill[key]) > max_len:
            max_len = len(d_fill[key])
    for key in d_fill.keys():
        if len(d_fill[key]) < max_len:
            for diff_val in xrange(max_len - len(d_fill[key])):
                d_fill[key].append(fill_val)
    return d_fill


def combine_marg_df():
    # outer_outdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\precip_perturbations"
    outer_outdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\precip_perturbations"
    existing_csv = [
        os.path.join(outer_outdir, f) for f in os.listdir(outer_outdir) if
        f.startswith('num_months_insufficient_summary')]
    existing_df = [pd.read_csv(csv) for csv in existing_csv]
    done_df = pd.concat(existing_df)
    done_df.drop_duplicates(inplace=True)
    now_str = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
    save_as = os.path.join(
        outer_outdir,
        'num_months_insufficient_summary_{}.csv'.format(now_str))
    done_df.to_csv(save_as, index=False)
    for item in existing_csv:
        os.remove(item)


def summarize_energy_balance(summary_csv, herbivore_csv, save_as):
    """Summarize energy and protein balance of the livestock herd."""
    herd_input = pd.read_csv(herbivore_csv)
    age_classes = herd_input.label.values.tolist()
    model_output = pd.read_csv(summary_csv)
    summary = model_output[['step']]
    for class_i in age_classes:
        protein_intake = model_output['{}DPLS'.format(class_i)]
        protein_req = (
            model_output['{}Pc'.format(class_i)] +
            model_output['{}Pl'.format(class_i)] +
            model_output['{}Pm'.format(class_i)])
        protein_ratio = protein_intake / protein_req
        e_intake = model_output['{}MEItotal'.format(class_i)]
        e_req = (
            model_output['{}MEc'.format(class_i)] +
            model_output['{}MEl'.format(class_i)] +
            model_output['{}MEm'.format(class_i)])
        e_ratio = e_intake / e_req
        summary['e_ratio_{}'.format(class_i)] = e_ratio
        summary['protein_ratio_{}'.format(class_i)] = protein_ratio
    summary.to_csv(save_as)


def max_viable_density_rainfall_perturbations():
    """Estimate maximum viable density with perturbed precip scenarios.

    For each perturbed precipitation scenario (or a subset), run the model
    at several stocking densities and calculate the number of months during
    the run where the herd experienced diet insufficiency.

    Returns:
        None
    """
    site_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_input_dir = r"C:\Users\ginge\Documents\NatCap\model_inputs_Kenya\regional_precip_perturbations"
    outer_outdir = r"C:/Users/ginge/Desktop/model_results_for_sarah"
    # r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\precip_perturbations"
    conception_step = -4
    template_herb_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\Ol_pej_herd.csv"
    herd_input = pd.read_csv(template_herb_csv)
    age_classes = herd_input.label.values.tolist()
    marg_dict = {
        'site': [],
        'density': [],
        'total_precip_perc_change': [],
        'PCI_perc_change': [],
        'num_months_insufficient': [],
    }
    # ------------------------------------------------------------------------
    # stuff to vary that controls the experiment
    # full change_perc_series:
    # [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
    change_perc_series = [0]  # [-0.8, -0.4, 0, 0.4, 0.8, 1.2]
    num_sites = 5
    num_tries = 3  # 50
    # this dict specifies starting stocking density according to the percent
    # change in mean precip
    starting_density_dict = {
        -0.8: 0.1,
        -0.6: 0.25,
        -0.4: 0.4,
        0: 1.1,
        0.4: 1.3,
        0.6: 1.4,
        0.8: 1.5,
        1.2: 2.,
    }
    # site_subset = np.random.choice(len(site_list), num_sites)
    # site_subset = [0, 1, 2, 5, 9, 11, 12, 24, 18, 21]  # 10 sites chosen to span range of back-calc densities
    site_subset = [4]  # [4, 14, 15, 16, 19, 22, 23, 8, 17]
    # ------------------------------------------------------------------------
    forage_args = default_forage_args()
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['template_level'] = 'GL'
    edit_conception_month(template_herb_csv, conception_step)
    target = 0.14734
    try:
        existing_csv = [
            os.path.join(outer_outdir, f) for f in os.listdir(outer_outdir) if
            f.startswith('num_months_insufficient_summary')]
        existing_df = [pd.read_csv(csv) for csv in existing_csv]
        done_df = pd.concat(existing_df)
        done_df.drop_duplicates(inplace=True)
        for mean_change_perc in [-0.6, 0.6]:  # change_perc_series:
            pci_change_perc = 0
            # for pci_change_perc in change_perc_series:
            input_dir = os.path.join(
                outer_input_dir, 'total_precip_{}_PCI_{}'.format(
                    mean_change_perc, pci_change_perc))
            forage_args['input_dir'] = os.path.join(outer_input_dir, input_dir)
            for site_index in xrange(len(site_list)):  # site_subset:
                site = site_list[site_index]
                # find scenarios already run for this site
                site_df = done_df.loc[
                    (done_df['site'] == site['name']) &
                    (done_df['total_precip_perc_change'] ==
                                mean_change_perc) &
                    (done_df['PCI_perc_change'] == pci_change_perc)]
                if len(site_df > 0):
                    if min(site_df['density']) < 0.00001:
                        continue  # give up
                    if (1 in site_df['num_months_insufficient'].values &
                            2 in site_df['num_months_insufficient'].values):
                        continue  # this combination is done
                    if max(site_df['num_months_insufficient']) == 0:
                        # start from the maximum density tested
                        density = max(site_df['density'].values)
                        num_months_insufficient = 0
                    else:
                        # start from density with smallest num months
                        # insufficient
                        pos_df = site_df.loc[
                            site_df['num_months_insufficient'] > 0]
                        num_months_insufficient = min(
                            pos_df['num_months_insufficient'])
                        density = pos_df.loc[
                            pos_df['num_months_insufficient'] ==
                            num_months_insufficient]['density']
                        if len(density) > 1:
                            density = float(max(density.values))
                        else:
                            density = float(density.values)
                else:  # no done runs for this combination yet
                    density = starting_density_dict[mean_change_perc]
                    num_months_insufficient = 0
                    calc_stocking_density(template_herb_csv, density)
                for i in xrange(num_tries + 1):
                    if num_months_insufficient >= 2:
                        if num_months_insufficient >= 10:
                            density = density - density * 0.35
                        else:
                            density = density - 0.0006
                        calc_stocking_density(template_herb_csv, density)
                    elif num_months_insufficient == 0:
                        density = density + 0.0001
                        calc_stocking_density(template_herb_csv, density)
                    else:
                        continue
                    outdir = os.path.join(
                        outer_outdir, 'exp{}'.format(site['name']))
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    grass_csv = os.path.join(
                        outer_input_dir, input_dir,
                        '{:d}.csv'.format(int(site['name'])))
                    add_cp_to_grass_csv(grass_csv, target)
                    forage_args['grass_csv'] = grass_csv
                    forage_args['latitude'] = site['lat']
                    forage_args['outdir'] = outdir
                    sum_csv = os.path.join(outdir, 'summary_results.csv')
                    forage.execute(forage_args)
                    model_output = pd.read_csv(sum_csv)
                    model_output['herd_avg_diet_sufficiency'] = 0
                    for class_i in age_classes:
                        class_proportion = herd_input.loc[
                            herd_input['label'] == class_i, 'proportion_of_herd']
                        energy_intake = model_output['{}MEItotal'.format(class_i)]
                        energy_req = model_output['{}E_req'.format(class_i)]
                        diet_sufficiency = (
                            (energy_intake - energy_req) / energy_req)
                        model_output['herd_avg_diet_sufficiency'] += (
                            diet_sufficiency * class_proportion.values)
                    num_months_insufficient = len(
                        model_output[
                            model_output['herd_avg_diet_sufficiency'] < 0])
                    marg_dict['site'].append(site['name'])
                    marg_dict['density'].append(density)
                    marg_dict['total_precip_perc_change'].append(
                        mean_change_perc)
                    marg_dict['PCI_perc_change'].append(
                        pci_change_perc)
                    marg_dict['num_months_insufficient'].append(
                        num_months_insufficient)
                    try:
                        shutil.rmtree(outdir)
                    except WindowsError:
                        pass
    finally:
        filled_dict = fill_dict(marg_dict, 'NA')
        marg_df = pd.DataFrame(filled_dict)

        now_str = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        save_as = os.path.join(
            outer_outdir,
            'num_months_insufficient_summary_{}.csv'.format(now_str))
        marg_df.to_csv(save_as, index=False)


def rainfall_perturbations_experiment():
    """Run the model at regional properties with perturbed precip.

    Run the model at one stocking density with precip that has been perturbed
    in terms of total amount (increased or decreased annual precip) or intra-
    annual variability, keeping annual precip constant.

    Returns:
        None
    """
    outer_input_dir = r"C:\Users\ginge\Documents\NatCap\model_inputs_Kenya\regional_precip_perturbations"
    tot_prec_regex = re.compile(r'total_precip_(.+?)_')
    pci_regex = re.compile(r'_PCI_(.+?$)')
    density = 1.015
    conception_step = -4

    template_herb_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\Ol_pej_herd.csv"
    herd_input = pd.read_csv(template_herb_csv)
    age_classes = herd_input.label.values.tolist()
    marg_dict = {
        'site': [], 'step': [], 'herd_avg_diet_sufficiency': [],
        'total_precip_perc_change': [], 'PCI_perc_change': []}
    site_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_outdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\precip_perturbations"
    forage_args = default_forage_args()
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['template_level'] = 'GL'
    edit_conception_month(template_herb_csv, conception_step)
    calc_stocking_density(template_herb_csv, density)
    target = 0.14734
    input_dir_list = [
        f for f in os.listdir(outer_input_dir)
        if os.path.isdir(os.path.join(outer_input_dir, f))]
    for input_dir in input_dir_list:
        tot_precip_perc_change = float(
            tot_prec_regex.search(input_dir).group(1))
        PCI_perc_change = float(pci_regex.search(input_dir).group(1))
        forage_args['input_dir'] = os.path.join(outer_input_dir, input_dir)
        for site in site_list:
            outdir = os.path.join(
                outer_outdir, '{}_total_precip_{}_PCI_{}'.format(
                    site['name'], tot_precip_perc_change, PCI_perc_change))
            grass_csv = os.path.join(
                outer_input_dir, input_dir,
                '{:d}.csv'.format(int(site['name'])))
            forage_args['grass_csv'] = grass_csv
            forage_args['latitude'] = site['lat']
            forage_args['outdir'] = outdir
            sum_csv = os.path.join(outdir, 'summary_results.csv')
            if not os.path.isfile(sum_csv):
                forage.execute(forage_args)
            model_output = pd.read_csv(sum_csv)
            model_output['herd_avg_diet_sufficiency'] = 0
            for class_i in age_classes:
                class_proportion = herd_input.loc[
                    herd_input['label'] == class_i, 'proportion_of_herd']
                energy_intake = model_output['{}MEItotal'.format(class_i)]
                energy_req = model_output['{}E_req'.format(class_i)]
                diet_sufficiency = (energy_intake - energy_req) / energy_req
                model_output['herd_avg_diet_sufficiency'] += (
                    diet_sufficiency * class_proportion.values)
            herd_avg_diet_sufficiency = model_output[
                'herd_avg_diet_sufficiency'].values.tolist()
            marg_dict['herd_avg_diet_sufficiency'].extend(
                herd_avg_diet_sufficiency)
            marg_dict['site'].extend(
                [site['name']] * len(herd_avg_diet_sufficiency))
            marg_dict['step'].extend(model_output['step'])
            marg_dict['total_precip_perc_change'].extend(
                [tot_precip_perc_change] * len(herd_avg_diet_sufficiency))
            marg_dict['PCI_perc_change'].extend(
                [PCI_perc_change] * len(herd_avg_diet_sufficiency))
    marg_df = pd.DataFrame(marg_dict)
    marg_df.to_csv(os.path.join(outer_outdir, 'energy_balance_summary.csv'))


def Laikipia_uniform_density():
    """Launch the beta model to mimic RPM."""
    template_herb_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herd_avg_uncalibrated.csv"
    template_grass_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015/0.csv"
    grass_df = pd.read_csv(template_grass_csv)
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/soil.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_results/regional_properties/uniform_density_9.17.2020"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_inputs"
    forage_args = default_forage_args()
    forage_args['input_dir'] = input_dir
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['template_level'] = 'GL'
    # target = 0.14734  # this is already in the grass_csv
    for site in site_list[:5]:
        grass_df.at[0, 'label'] = '{:d}'.format(int(site['id']))
        temp_grass_csv_path = 'C:/Users/ginge/Desktop/{:d}.csv'.format(
            int(site['id']))
        grass_df.to_csv(temp_grass_csv_path)
        outdir = os.path.join(outer_outdir, 'site_{}'.format(int(site['id'])))
        forage_args['grass_csv'] = temp_grass_csv_path
        forage_args['latitude'] = site['latitude']
        forage_args['outdir'] = outdir
        forage.execute(forage_args)
        os.remove(temp_grass_csv_path)


def Kenya_MS_regional_productivity():
    """Run the forage model at constant stocking density.

    Launch the model for each regional property at a constant stocking density.
    Generate results for Kenya MS to demonstrate regional productivity.
    This code adapted from the function run_preset_densities(), Sept 2018.

    Returns:
        None.
    """
    template_herb_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\Ol_pej_herd.csv"
    herd_input = pd.read_csv(template_herb_csv)
    age_classes = herd_input.label.values.tolist()
    min_biomass_dict = {
        'site': [], 'density': [], 'conception_step': [], 'min_biomass': []}
    marg_dict = {
        'site': [], 'density': [], 'step': [], 'herd_avg_diet_sufficiency': [],
        'conception_step': []}
    site_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_outdir = r"C:\Users\ginge\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\energy_balance"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015"
    forage_args = default_forage_args()
    forage_args['input_dir'] = input_dir
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['template_level'] = 'GL'
    target = 0.14734
    density_list = [0.07, 0.385, 1.33, 1.015, 0.7]  # cattle per ha
    conception_step_list = [-12, -8, -4, 0, 4, 8]
    for conception_step in conception_step_list:
        edit_conception_month(template_herb_csv, conception_step)
        for density in density_list:
            for site in site_list:
                outdir = os.path.join(
                    outer_outdir, 'site_{:d}_{}_{}'.format(int(site['name']),
                    density, conception_step))
                calc_stocking_density(template_herb_csv, density)
                grass_csv = os.path.join(input_dir,
                                         '{:d}.csv'.format(int(site['name'])))
                # add_cp_to_grass_csv(grass_csv, target)  # TODO fluctuating N content?
                forage_args['grass_csv'] = grass_csv
                forage_args['latitude'] = site['lat']
                forage_args['outdir'] = outdir
                sum_csv = os.path.join(outdir, 'summary_results.csv')
                if not os.path.isfile(sum_csv):
                    forage.execute(forage_args)
                model_output = pd.read_csv(sum_csv)
                model_output['herd_avg_diet_sufficiency'] = 0
                for class_i in age_classes:
                    class_proportion = herd_input.loc[
                        herd_input['label'] == class_i, 'proportion_of_herd']
                    energy_intake = model_output['{}MEItotal'.format(class_i)]
                    energy_req = model_output['{}E_req'.format(class_i)]
                    diet_sufficiency = (energy_intake - energy_req) / energy_req
                    model_output['herd_avg_diet_sufficiency'] += (
                        diet_sufficiency * class_proportion.values)
                herd_avg_diet_sufficiency = model_output[
                    'herd_avg_diet_sufficiency'].values.tolist()
                marg_dict['herd_avg_diet_sufficiency'].extend(
                    herd_avg_diet_sufficiency)
                marg_dict['site'].extend(
                    [site['name']] * len(herd_avg_diet_sufficiency))
                marg_dict['density'].extend(
                    [density] * len(herd_avg_diet_sufficiency))
                marg_dict['step'].extend(model_output['step'])
                marg_dict['conception_step'].extend(
                    [conception_step] * len(herd_avg_diet_sufficiency))

                total_biomass = (
                    model_output['{}_dead_kgha'.format(int(site['name']))] +
                    model_output['{}_green_kgha'.format(int(site['name']))])
                min_biomass = min(total_biomass)
                min_biomass_dict['min_biomass'].append(min_biomass)
                min_biomass_dict['site'].append(site['name'])
                min_biomass_dict['density'].append(density)
                min_biomass_dict['conception_step'].append(conception_step)
    marg_df = pd.DataFrame(marg_dict)
    marg_df.to_csv(os.path.join(outer_outdir, 'energy_balance_summary.csv'))

    min_biomass_df = pd.DataFrame(min_biomass_dict)
    min_biomass_df.to_csv(os.path.join(outer_outdir, 'min_biomass_summary.csv'))


def run_preset_densities():
    """Run a series of stocking densities at each regional property."""

    failed = []
    template_herb_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\herd_avg_uncalibrated.csv"
    # "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/castrate_suyian.csv"
    density_list = [0.07, 0.385, 0.7, 1.015, 1.33]  # cattle per ha
    target = 0.14734

    marg_dict = {'site': [], 'density': [], 'avg_yearly_gain': [],
                 'total_yearly_delta_weight_kg_per_ha': []}
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\herd_avg_uncalibrated_varying_cp_GL_est_densities"  # herd_avg_uncalibrated_constant_cp_GL" # herd_avg_uncalibrated_0.3_vary_cp"
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015"
    forage_args = default_forage_args()
    forage_args['user_define_protein'] = 1
    forage_args['input_dir'] = input_dir
    forage_args['herbivore_csv'] = template_herb_csv
    forage_args['restart_monthly'] = 1
    forage_args['template_level'] = 'GL'
    # for density in density_list:
    for site in site_list:
        density = site['back_calc_avg_animals_per_ha']
        # density = 0.3  # mean reported density across properties
        outdir = os.path.join(outer_outdir,
                              'site_{:d}_{}'.format(int(site['name']), density))
        modify_stocking_density(template_herb_csv, density)
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        initialize_n_mult(grass_csv)
        # add_cp_to_grass_csv(grass_csv, target)
        forage_args['grass_csv'] = grass_csv
        forage_args['latitude'] = site['lat']
        forage_args['outdir'] = outdir
        # if not succeeded:  # os.path.exists(outdir):
        if density > 0:
            calc_n_mult(forage_args, target)
        # try:
        forage.execute(forage_args)
        # except:
            # import pdb; pdb.set_trace()
            # continue
        succeeded = id_failed_simulation(outdir, forage_args['num_months'])
        if not succeeded:
            failed.append('site_{:d}_{}_per_ha'.format(int(site['name']),
                          density))
        # else:
        sum_csv = os.path.join(outdir, 'summary_results.csv')
        try:
            sum_df = pd.read_csv(sum_csv)
        except:
            continue
        subset = sum_df.loc[sum_df['year'] > 2013]
        grouped = subset.groupby('year')
        avg_yearly_gain = (grouped.sum()['cattle_gain_kg']).mean()
        start_wt = sum_df.iloc[0]['cattle_kg']
        avg_yearly_gain_herd = avg_yearly_gain * float(density)
        perc_gain = avg_yearly_gain / float(start_wt)
        marg_dict['site'].append(site['name'])
        marg_dict['density'].append(density)
        marg_dict['avg_yearly_gain'].append(avg_yearly_gain)
        marg_dict['total_yearly_delta_weight_kg_per_ha'].append(
                                                      avg_yearly_gain_herd)
    # summarize_cp_content(outer_outdir)
    df = pd.DataFrame(marg_dict)
    summary_csv = os.path.join(outer_outdir, 'gain_summary.csv')
    df.to_csv(summary_csv)
    erase_intermediate_files(outer_outdir)
    if len(failed) > 0:
        print("the following sites failed:")
        print(failed)

def erase_intermediate_files(outerdir):
    for folder in os.listdir(outerdir):
        try:
            for file in os.listdir(os.path.join(outerdir, folder)):
                if file.endswith("summary_results.csv") or \
                file.startswith("forage-log") or \
                file.endswith("summary.csv"):
                    continue
                else:
                    try:
                        object = os.path.join(outerdir, folder, file)
                        if os.path.isfile(object):
                            os.remove(object)
                        else:
                            shutil.rmtree(object)
                    except OSError:
                        continue
        except WindowsError:
            continue

def summarize_match(match_csv, save_as):
    """summarize the results of back-calc management routine by collecting
    the final comparison of empirical vs simulated biomass from each site
    where the routine was run."""

    sum_dict = {'year': [], 'site': [], 'live_or_total': [], 'g_m2': [],
                'sim_vs_emp': []}
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014"
    for live_or_total in ['total']:  # , 'live']:
        for year_to_match in [2015]:  # , 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            for site in site_list:
                site_name = site['name']
                # site_dir = os.path.join(outer_outdir,
                                        # 'back_calc_{}_{}'.format(year_to_match,
                                        # live_or_total),
                                        # 'FID_{}'.format(site_name))
                site_dir = os.path.join(outer_outdir,
                                        'back_calc_match 2015',
                                        'FID_{}'.format(site_name))
                result_csv = os.path.join(site_dir,
                                          'modify_management_summary_{}.csv'.
                                          format(site_name))
                res_df = pd.read_csv(result_csv)
                sum_dict['year'].extend([year_to_match] * 2)
                sum_dict['site'].extend([site_name] * 2)
                sum_dict['live_or_total'].extend([live_or_total] * 2)
                sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                        Simulated_biomass)
                sum_dict['sim_vs_emp'].append('sim')
                sum_dict['g_m2'].append(res_df.iloc[len(res_df) - 1].
                                        Empirical_biomass)
                sum_dict['sim_vs_emp'].append('emp')
    sum_df = pd.DataFrame(sum_dict)
    sum_df.to_csv(save_as)

def set_grass_cp(grass_csv, live_cp, dead_cp):
    """Assign fixed crude protein, reset N multiplier"""

    grass_df = pd.read_csv(grass_csv)
    grass_df['index'] = 0
    grass_df = grass_df.set_index(['index'])
    assert len(grass_df) == 1, "We can only handle one grass type"
    grass_df['cprotein_green'] = grass_df['cprotein_green'].astype(float)
    grass_df.set_value(0, 'cprotein_green', live_cp)
    grass_df['cprotein_dead'] = grass_df['cprotein_dead'].astype(float)
    grass_df.set_value(0, 'cprotein_dead', dead_cp)
    grass_df.set_value(0, 'N_multiplier', 1)
    grass_df.to_csv(grass_csv)

def calc_n_mult(forage_args, target):
    """Calculate N multiplier for a grass to achieve target crude protein
    content, and edit grass input to include that N multiplier. Target reflects
    crude protein of live grass. Target should be supplied as a float between 0
    and 1."""

    tolerance = 0.001  # must be within this proportion of target value
    grass_df = pd.read_csv(forage_args['grass_csv'])
    grass_label = grass_df.iloc[0].label
    # args copy to launch model to calculate n_mult
    args_copy = forage_args.copy()
    args_copy['outdir'] = os.path.join(os.path.dirname(forage_args['outdir']),
                                       '{}_n_mult_calc'.format(
                                      os.path.basename(forage_args['outdir'])))
    # find correct output time period
    final_month = forage_args[u'start_month'] + forage_args['num_months'] - 1
    if final_month > 12:
        mod = final_month % 12
        if mod == 0:
            month = 12
            year = (final_month / 12) + forage_args[u'start_year'] - 1
        else:
            month = mod
            year = (final_month / 12) + forage_args[u'start_year']
    else:
        month = final_month
        year = ((forage_args['num_months'] - 1) / 12) + forage_args[u'start_year']
    intermediate_dir = os.path.join(args_copy['outdir'],
                                    'CENTURY_outputs_m%d_y%d' % (month, year))
    sim_output = os.path.join(intermediate_dir, '{}.lis'.format(grass_label))
    first_year = forage_args['start_year']
    last_year = year

    def get_raw_cp_green():
        # calculate n multiplier to achieve target
        outputs = cent.read_CENTURY_outputs(sim_output, first_year, last_year)
        outputs.drop_duplicates(inplace=True)

        # restrict to months of the simulation
        first_month = forage_args[u'start_month']
        start_date = first_year + float('%.2f' % (first_month / 12.))
        end_date = last_year + float('%.2f' % (month / 12.))
        outputs = outputs[(outputs.index >= start_date)]
        outputs = outputs[(outputs.index <= end_date)]
        return np.mean(outputs.aglive1 / outputs.aglivc)

    def set_n_mult():
        # edit grass csv to reflect calculated n_mult
        grass_df = pd.read_csv(forage_args['grass_csv'])
        grass_df.N_multiplier = grass_df.N_multiplier.astype(float)
        grass_df = grass_df.set_value(0, 'N_multiplier', float(n_mult))
        grass_df = grass_df.set_index('label')
        grass_df.to_csv(forage_args['grass_csv'])

    n_mult = 1
    set_n_mult()
    forage.execute(args_copy)
    cp_green = get_raw_cp_green()
    diff = abs(target - (n_mult * cp_green))
    while diff > tolerance:
        n_mult = '%.10f' % (target / cp_green)
        set_n_mult()
        forage.execute(args_copy)
        cp_green = get_raw_cp_green()
        diff = abs(target - (float(n_mult) * cp_green))

def calc_n_months(match_csv, site_name):
    """Calculate the number of months to run the model, from 2014 measurement
    to 2015 measurement."""

    site_df = pd.read_csv(match_csv)

    sub_2015 = site_df.loc[(site_df.FID == int(site_name)) &
                           (site_df.Year == 2015)]
    date_list = sub_2015.sim_date.values[0].rsplit("/")
    year = date_list[2]
    month = '0{}'.format(date_list[0])
    date_2015 = pd.to_datetime('{}-{}-28'.format(year, month),
                               format='%Y-%m-%d')

    sub_2014 = site_df.loc[(site_df.FID == int(site_name)) &
                           (site_df.Year == 2014)]
    date_list = sub_2014.sim_date.values[0].rsplit("/")
    year = date_list[2]
    month = '0{}'.format(date_list[0])
    date_2014 = pd.to_datetime('{}-{}-28'.format(year, month),
                               format='%Y-%m-%d')
    return (date_2015 - date_2014).days / 30

def move_input_files():
    """move the final schedule and graz files (if exists) from back-calc
    management output to a folder of inputs to run forward from a back-calc
    management result."""

    back_calc_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\back_calc_2014_total"
    orig_input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    new_input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\forward_from_2014"

    folders = [f for f in os.listdir(back_calc_dir) if
               os.path.isdir(os.path.join(back_calc_dir, f))]
    for folder in folders:
        site_dir = os.path.join(back_calc_dir, folder)
        FID = folder[4:]
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [int(re.search('{}_{}(.+?).sch'.format(FID,
                         FID), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:  # no schedule modification was needed
            final_sch = os.path.join(orig_input_dir, '{}.sch'.format(FID))
        else:
            final_sch_iter = max(sch_iter_list)
            final_sch = os.path.join(site_dir, '{}_{}{}.sch'.format(FID,
                                     FID, final_sch_iter))
        grz_files = [f for f in os.listdir(site_dir) if f.startswith('graz')]
        if len(grz_files) > 0:
            grz_iter_list = [int(re.search('graz_{}(.+?).100'.format(
                             FID), f).group(1)) for f in grz_files]
            final_iter = max(grz_iter_list)
            final_grz = os.path.join(site_dir, 'graz_{}{}.100'.format(
                                     FID, final_iter))
            shutil.copyfile(final_grz, os.path.join(new_input_dir,
                                                    'graz_{}.100'.format(
                                                     FID)))
        shutil.copyfile(final_sch, os.path.join(new_input_dir, '{}.sch'.format(FID)))

def remove_grazing(match_csv, sch_dir):
    """This was a good start but turned out to be harder than I thought so I
    abandoned it!!  The hard part is target_dict: it looks for the month
    previous to modify, but we  want to modify the month after."""
    """Remove grazing from schedule files in the sch_dir after the date for
    which biomass was matched (2014).  We assume that the correct (final)
    schedule file was already moved to the sch_dir using the function
    move_input_files()."""

    site_list = generate_inputs(match_csv, 2014, 'total')
    for site in site_list:
        measurement_date = site['date']
        date = measurement_date + 0.17
        sch_file = os.path.join(sch_dir, '{}.sch'.format(site['name']))
        target_dict = cent.find_target_month(0, sch_file, date, 12)

        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as new_file:
            with open(sch_file, 'rb') as sch:
                for line in sch:
                    if 'Last year' in line:
                        year = int(line[:9].strip())
                        if year == int(target_dict['last_year']):
                            # this is the target block
                            new_file.write(line)
                            prev_event_month_count = 0
                            while '-999' not in line:
                                line = sch.next()
                                if 'Labeling type' in line:
                                    new_file.write(line)
                                    break
                                if line[:3] == "   ":
                                    year = int(line[:5].strip())
                                    if year == int(target_dict['target_year']):
                                        month = int(line[7:10].strip())
                                        if month >= target_dict['target_month']:
                                            if line[10:].strip() == 'GRAZ':
                                                sch.next()
                                                continue
                                    if year > int(target_dict['target_year']):
                                        if line[10:].strip() == 'GRAZ':
                                            sch.next()
                                            continue
                                new_file.write(line)
                            line = sch.next()
                    new_file.write(line)

        # just in case TODO copy abs_path to sch_file
        new_sch = os.path.join(sch_dir, '{}_modified.sch'.format(site['name']))
        shutil.copyfile(abs_path, new_sch)
        os.remove(abs_path)

def multiply_herb_densities(herb_csv, density_mult):
    """Create a new herbivore csv for input to the model with stocking density
    values multiplied by a constant, density_mult. return the filepath to the
    new csv."""

    throwaway_dir = "C:/Users/Ginger/Desktop/throwaway_dir"
    if not os.path.exists(throwaway_dir):
        os.makedirs(throwaway_dir)
    herb_df = pd.read_csv(herb_csv)
    herb_df = herb_df.set_index(['label'])
    herb_df['stocking_density'] = herb_df['stocking_density'] * float(density_mult)
    new_path = os.path.join(throwaway_dir, os.path.basename(herb_csv))
    herb_df.to_csv(new_path)
    return new_path

def empirical_densities_forward(match_csv, empir_outdir, density_mult):
    """Run forward from schedule matching biomass in 2014, with empirical
    densities of different animal types."""

    live_cp = 0.1473
    dead_cp = 0.0609  # averages from lit review

    # compare these results to those from back_calc_forward
    forage_args = default_forage_args()
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_forward_from_2014"
    herb_csv_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\by_property"
    site_list = generate_inputs(match_csv, 2014, 'total')
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    lat_df = pd.read_csv(site_csv)

    # move inputs (graz and schedule file) from back-calc results directory
    # to this new input directory
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\empty_after_2014"

    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    fix_file = 'drytrpfi.100'
    forage_args['input_dir'] = input_dir
    for site in site_list:
        if site['name'] == '12':
            continue  # skip Lombala
        # find graz file associated with back-calc management
        graz_filter = [f for f in os.listdir(input_dir) if
                       f.startswith('graz_{}'.format(site['name']))]
        if len(graz_filter) == 1:
            graz_file = os.path.join(input_dir, graz_filter[0])
            def_graz_file = os.path.join(century_dir, 'graz.100')
            shutil.copyfile(def_graz_file, os.path.join(century_dir,
                            'default_graz.100'))
            shutil.copyfile(graz_file, def_graz_file)
        # find herbivore input for this site
        herb_csv = os.path.join(herb_csv_dir, '{}.csv'.format(site['name']))
        # multiply by constant density multiplier
        herb_csv_mult = multiply_herb_densities(herb_csv, density_mult)
        forage_args['herbivore_csv'] = herb_csv_mult

        # modify crude protein of grass for this site, set N_mult to 1
        grass_csv = os.path.join(input_dir, '{}.csv'.format(site['name']))
        set_grass_cp(grass_csv, live_cp, dead_cp)
        forage_args['grass_csv'] = grass_csv

        # calculate n_months to run as difference between
        # 2014 and 2015 measurements
        n_months = calc_n_months(match_csv, site['name'])
        forage_args['num_months'] = n_months
        forage_args['start_year'] = 2014
        mo_float = (site['date'] - 2014) * 12.0
        if mo_float - int(mo_float) > 0.5:
            month = int(mo_float) + 2
        else:
            month = int(mo_float) + 1
        forage_args['start_month'] = month
        out_dir_site = os.path.join(empir_outdir, '{}'.format(site['name']))
        forage_args['outdir'] = out_dir_site
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site)
            forage_args['latitude'] = (lat_df[lat_df.name == int(site['name'])].lat).tolist()[0]
            forage.execute(forage_args)

        if len(graz_filter) == 1:
            shutil.copyfile(os.path.join(century_dir, 'default_graz.100'),
                            def_graz_file)
            os.remove(os.path.join(century_dir, 'default_graz.100'))

def compare_biomass(match_csv, empir_outdir, density_mult):
    """collect biomass from empirical densities, and back-calc schedules."""

    bc_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014\back_calc_match 2015"

    sum_dict = {'site': [], 'biomass_emp_densities': [],
                'biomass_back_calc': [], 'density_multiplier': []}
    site_list = generate_inputs(match_csv, 2015, 'total')
    for site in site_list:
        if site['name'] == '12':
            continue  # skip Lombala
        match_year = 2015
        mo_float = (site['date'] - 2015) * 12.0
        if mo_float - int(mo_float) > 0.5:
            match_month = int(mo_float) + 1
        else:
            match_month = int(mo_float)
        emp_res = pd.read_csv(os.path.join(empir_outdir, site['name'],
                                           'summary_results.csv'))
        emp_subs = emp_res.loc[(emp_res.year == match_year) &
                               (emp_res.month == match_month)]
        if emp_subs.shape[0] != 1:
            import pdb; pdb.set_trace()
        # assert emp_subs.shape[0] == 1, "must be one row"
        emp_biomass = (emp_subs.iloc[0]['{}_dead_kgha'.format(site['name'])] +
                       emp_subs.iloc[0]['{}_green_kgha'.format(site['name'])]) * 0.1
        # TODO green biomass? later
        bc_res = pd.read_csv(os.path.join(bc_outdir,
                                          'FID_{}'.format(site['name']),
                                          'modify_management_summary_{}.csv'.format(site['name'])))
        max_iter = np.max(bc_res.Iteration)
        bc_biomass = bc_res.iloc[max_iter].Simulated_biomass
        sum_dict['site'].append(site['name'])
        sum_dict['biomass_emp_densities'].append(emp_biomass)
        sum_dict['biomass_back_calc'].append(bc_biomass)
        sum_dict['density_multiplier'].append(density_mult)
    sum_df = pd.DataFrame(sum_dict)
    sum_df.set_index('site', inplace=True)
    return sum_df

def back_calc_forward(match_csv, template_level):
    """Run the model forward from a back-calculated schedule, and match biomass
    at a later date."""

    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014"

    live_or_total = 'total'
    year_to_match = 2015

    site_list = generate_inputs(match_csv, year_to_match, live_or_total)

    # move inputs (graz and schedule file) from back-calc results directory
    # to this new input directory
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\forward_from_2014"

    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    out_dir = os.path.join(outer_outdir, "back_calc_match {}".format(
                                                                year_to_match))
    vary = 'both'
    threshold = 10.0
    max_iterations = 40
    fix_file = 'drytrpfi.100'

    for site in site_list:
        # find graz file associated with back-calc management
        graz_filter = [f for f in os.listdir(input_dir) if
                       f.startswith('graz_{}'.format(site['name']))]
        if len(graz_filter) == 1:
            graz_file = os.path.join(input_dir, graz_filter[0])
            def_graz_file = os.path.join(century_dir, 'graz.100')
            shutil.copyfile(def_graz_file, os.path.join(century_dir,
                            'default_graz.100'))
            shutil.copyfile(graz_file, def_graz_file)

        # calculate n_months to back-calc management as difference between
        # 2014 and 2015 measurements
        n_months = calc_n_months(match_csv, site['name'])

        out_dir_site = os.path.join(out_dir, 'FID_{}'.format(site['name']))
        if not os.path.exists(out_dir_site):
            os.makedirs(out_dir_site)
        backcalc.back_calculate_management(site, input_dir,
                                           century_dir, out_dir_site,
                                           fix_file, n_months,
                                           vary, live_or_total,
                                           threshold, max_iterations,
                                           template_level)
        if len(graz_filter) == 1:
            shutil.copyfile(os.path.join(century_dir, 'default_graz.100'),
                            def_graz_file)
            os.remove(os.path.join(century_dir, 'default_graz.100'))

def back_calc_mgmt(match_csv, template_level):
    """Use the back-calc management routine to calculate management at regional
    properties prior to the 2014 or 2015 measurement."""

    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties"
    for live_or_total in ['total']:  # , 'live']:
        for year_to_match in [2015]:  # , 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
            n_months = 24
            century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
            out_dir = os.path.join(outer_outdir, "back_calc_{}_{}".format(
                                                 year_to_match, live_or_total))
            vary = 'both'
            threshold = 10.0
            max_iterations = 40
            fix_file = 'drytrpfi.100'
            for site in site_list:
                out_dir_site = os.path.join(out_dir, 'FID_{}'.format(
                                                                 site['name']))
                if not os.path.exists(out_dir_site):
                    os.makedirs(out_dir_site)
                backcalc.back_calculate_management(site, input_dir,
                                                   century_dir, out_dir_site,
                                                   fix_file, n_months,
                                                   vary, live_or_total,
                                                   threshold, max_iterations,
                                                   template_level)

def generate_inputs(match_csv, year_to_match, live_or_total):
    """Generate a list that can be used as input to run the back-calc
    management routine.  Year_to_match should be 2014 or 2015.  live_or_total
    should be 'live' or 'total'."""

    site_list = []
    site_df = pd.read_csv(match_csv)
    for site in site_df.Property.unique():
        sub_df = site_df.loc[(site_df.Property == site) &
                             (site_df.Year == year_to_match)]
        assert len(sub_df) < 2, "must be only one site record to match"
        if len(sub_df) == 0:
            continue
        site_name = str(sub_df.get_value(sub_df.index[0], 'FID'))
        date_list = sub_df.sim_date.values[0].rsplit("/")
        year = date_list[2]
        month = date_list[0]
        century_date = round((int(year) + int(month) / 12.0), 2)
        site_dict = {'name': site_name, 'date': century_date}
        if live_or_total == 'live':
            site_dict['biomass'] = sub_df.get_value(sub_df.index[0],
                                                    'GBiomass')
        elif live_or_total == 'total':
            site_dict['biomass'] = sub_df.get_value(sub_df.index[0],
                                                    'mean_biomass_gm2')
        site_list.append(site_dict)
    return site_list

def run_baseline(site_csv):
    """Run the model with zero grazing, for each regional property."""

    forage_args = default_forage_args()
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip"
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        outdir = os.path.join(outer_output_dir,
                              '{:d}'.format(int(site['name'])))
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        forage_args['latitude'] = site['lat']
        forage_args['outdir'] = outdir
        forage_args['grass_csv'] = grass_csv
        forage_args['herbivore_csv'] = None
        forage_args['input_dir'] = input_dir
        forage.execute(forage_args)

def combine_summary_files(site_csv):
    """Make a file that can be used to plot biomass differences between
    sites."""

    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip\combined_summary.csv"
    df_list = []
    outer_output_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\zero_dens\Worldclim_precip"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        sim_name = int(site['name'])
        sim_dir = os.path.join(outer_output_dir, '{}'.format(sim_name))
        sim_df = pd.read_csv(os.path.join(sim_dir,'summary_results.csv'))
        sim_df['total_kgha'] = sim_df['{}_green_kgha'.format(sim_name)] + \
                                    sim_df['{}_dead_kgha'.format(sim_name)]
        sim_df['site'] = sim_name
        sim_df = sim_df[['site', 'total_kgha', 'month', 'year',
                         'total_offtake']]
        df_list.append(sim_df)
    combined_df = pd.concat(df_list)
    combined_df.to_csv(save_as)

def summarize_sch_wrapper(match_csv):
    """Wrapper function to summarize back-calculated schedules in several
    directories specified by year_to_match and live_or_total."""

    n_months = 24
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\forward_from_2014"
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014"
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    for live_or_total in ['total']:  # , 'live']:
        for year_to_match in [2015]:  # , 2015]:
            site_list = generate_inputs(match_csv, year_to_match,
                                        live_or_total)
            # outerdir = os.path.join(outer_outdir, "back_calc_{}_{}".format(
                                                 # year_to_match, live_or_total))
            outerdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014\back_calc_match 2015"
            raw_file = os.path.join(outerdir,
                                   "{}_{}_schedule_summary.csv".format(
                                   year_to_match, live_or_total))
            summary_file = os.path.join(outerdir,
                                   "{}_{}_percent_removed.csv".format(
                                   year_to_match, live_or_total))
            backcalc.summarize_calc_schedules(site_list, n_months, input_dir,
                                              century_dir, outerdir, raw_file,
                                              summary_file)

def add_cp_to_grass_csv(csv_file, target):
    """Modify the crude protein content in the grass csv used as input to the
    forage model."""

    df = pd.read_csv(csv_file)
    df['index'] = 0
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one grass type"
    df['cprotein_green'] = df['cprotein_green'].astype(float)
    df.set_value(0, 'cprotein_green', target)
    df['cprotein_dead'] = df['cprotein_dead'].astype(float)
    df.set_value(0, 'cprotein_dead', (target / 10.0))
    df.to_csv(csv_file)

def initialize_n_mult(csv_file):
    """Set N_multiplier to 1."""

    df = pd.read_csv(csv_file)
    df['index'] = 0
    df = df.set_index(['index'])
    assert len(df) == 1, "We can only handle one grass type"
    df.set_value(0, 'N_multiplier', 1)
    df.to_csv(csv_file)

def summarize_cp_content(outer_dir):
    """What was the cp content of grasses that was created with n_mult?"""

    cp_summary = {'site': [], 'n_mult': [], 'cp_mean': [], 'cp_stdev': []}
    input_dir = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015"
    forage_args = default_forage_args()
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    site_list = pd.read_csv(site_csv).to_dict(orient="records")
    for site in site_list:
        # get n_mult that was used
        grass_csv = os.path.join(input_dir,
                                 '{:d}.csv'.format(int(site['name'])))
        grass_df = pd.read_csv(grass_csv)
        grass_df['index'] = 0
        grass_df = grass_df.set_index(['index'])
        assert len(grass_df) == 1, "We can only handle one grass type"
        n_mult = grass_df.get_value(0, 'N_multiplier')

        # calculate cp content that was achieved
        outdir = os.path.join(outer_dir, 'site_{:d}'.format(int(site['name'])))
        final_month = forage_args[u'start_month'] + forage_args['num_months'] - 1
        if final_month > 12:
            mod = final_month % 12
            if mod == 0:
                month = 12
                year = (final_month / 12) + forage_args[u'start_year'] - 1
            else:
                month = mod
                year = (final_month / 12) + forage_args[u'start_year']
        else:
            month = final_month
            year = (step / 12) + forage_args[u'start_year']
        intermediate_dir = os.path.join(outdir,
                                        'CENTURY_outputs_m%d_y%d' %
                                        (month, year))
        grass_label = grass_df.iloc[0].label
        sim_output = os.path.join(intermediate_dir,
                                  '{}.lis'.format(grass_label))
        first_year = forage_args['start_year']
        last_year = year
        outputs = cent.read_CENTURY_outputs(sim_output, first_year, last_year)
        outputs.drop_duplicates(inplace=True)
        outputs.cp_green = (outputs.aglive1 / outputs.aglivc) * n_mult
        mean_cp_green = np.mean(outputs.cp_green)
        stdev_cp_green = np.std(outputs.cp_green)

        cp_summary['site'].append(site['name'])
        cp_summary['n_mult'].append(n_mult)
        cp_summary['cp_mean'].append(mean_cp_green)
        cp_summary['cp_stdev'].append(stdev_cp_green)
    sum_df = pd.DataFrame(cp_summary)
    sum_df.set_index('site', inplace =True)
    save_as = os.path.join(outer_dir, 'cp_summary.csv')
    sum_df.to_csv(save_as)

def back_calc_workflow():
    """Functions that were called under main when I was doing back-calc runs on
    regional properties."""

    # site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    # run_baseline(site_csv)
    # combine_summary_files(site_csv)
    # match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\regional_PDM_summary.csv"
    # back_calc_mgmt(match_csv)
    # move_input_files()
    # back_calc_forward(match_csv, 'GLPC')
    # summarize_sch_wrapper(match_csv)
    # save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_properties\forward_from_2014\back_calc_match_summary_2015.csv"
    # summarize_match(match_csv, save_as)
    # run_preset_densities()
    # summarize_offtake()
    summarize_remaining_biomass()

def back_calc_regional_avg():

    # back-calculate to match average biomass on regional properties in 2014
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\back_calc_match_2014"
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\back_calc_match_2014"
    fix_file = 'drytrpfi.100'
    n_months = 24
    vary = 'both'
    live_or_total = 'total'
    threshold = 10.0
    max_iterations = 40
    template_level = 'GH'
    site = {'biomass': 248.7, 'date': 2014.58, 'name': 'prop_avg'}
    # if not os.path.exists(out_dir):
        # os.makedirs(out_dir)
    # backcalc.back_calculate_management(site, input_dir,
                                       # century_dir, out_dir,
                                       # fix_file, n_months,
                                       # vary, live_or_total,
                                       # threshold, max_iterations,
                                       # template_level)

    # back-calculate forward to match average biomass on regional properties in 2015
    century_dir = 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\back_calc_match_2014"
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\back_calc_match_2015"
    fix_file = 'drytrpfi.100'
    n_months = 12
    vary = 'both'
    live_or_total = 'total'
    threshold = 10.0
    max_iterations = 40
    template_level = 'GH'
    site = {'biomass': 218.33, 'date': 2015.58, 'name': 'prop_avg'}
    # if not os.path.exists(out_dir):
        # os.makedirs(out_dir)
    # backcalc.back_calculate_management(site, input_dir,
                                       # century_dir, out_dir,
                                       # fix_file, n_months,
                                       # vary, live_or_total,
                                       # threshold, max_iterations,
                                       # template_level)
    save_raw = os.path.join(out_dir, "raw_summary.csv")
    save_summary = os.path.join(out_dir, "schedule_summary.csv")
    n_months = 13  # to go back to time when it should match scenario runs
    backcalc.summarize_calc_schedules([site], n_months, input_dir, century_dir,
                                      out_dir, save_raw, save_summary)

def scenario_mean_ecolclass_by_property(match_csv, empir_outdir, save_as):
    """Run each property forward from schedule matching biomass in 2014, at
    mean animal densities for each ecolclass. This is a mishmash of
    regional_scenarios() and empirical_densities_forward()"""

    live_cp = 0.1473
    dead_cp = 0.0609  # averages from lit review

    # compare these results to those from back_calc_forward
    forage_args = default_forage_args()
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_forward_from_2014"
    site_list = generate_inputs(match_csv, 2014, 'total')
    site_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\regional_properties.csv"
    lat_df = pd.read_csv(site_csv)

    # move inputs (graz and schedule file) from back-calc results directory
    # to this new input directory
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\empty_after_2014"

    century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
    fix_file = 'drytrpfi.100'
    forage_args['input_dir'] = input_dir
    for site in site_list:
        if site['name'] == '12':
            continue  # skip Lombala
        # find graz file associated with back-calc management
        graz_filter = [f for f in os.listdir(input_dir) if
                       f.startswith('graz_{}'.format(site['name']))]
        if len(graz_filter) == 1:
            graz_file = os.path.join(input_dir, graz_filter[0])
            def_graz_file = os.path.join(century_dir, 'graz.100')
            shutil.copyfile(def_graz_file, os.path.join(century_dir,
                            'default_graz.100'))
            shutil.copyfile(graz_file, def_graz_file)

        # modify crude protein of grass for this site, set N_mult to 1
        grass_csv = os.path.join(input_dir, '{}.csv'.format(site['name']))
        set_grass_cp(grass_csv, live_cp, dead_cp)
        forage_args['grass_csv'] = grass_csv

        # calculate n_months to run as difference between
        # 2014 and 2015 measurements
        n_months = calc_n_months(match_csv, site['name'])
        forage_args['num_months'] = n_months
        forage_args['start_year'] = 2014
        mo_float = (site['date'] - 2014) * 12.0
        if mo_float - int(mo_float) > 0.5:
            month = int(mo_float) + 2
        else:
            month = int(mo_float) + 1
        forage_args['start_month'] = month

        # run with animal densities reflecting each ecol class
        for ecol_class in ['livestock', 'integrated', 'wildlife']:
            forage_args['herbivore_csv'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\herbivores_regional_scenarios_{}.csv".format(ecol_class)
            out_dir_site = os.path.join(empir_outdir,
                                        '{}_{}'.format(site['name'],
                                                       ecol_class))
            forage_args['outdir'] = out_dir_site
            if not os.path.exists(out_dir_site):
                os.makedirs(out_dir_site)
                forage_args['latitude'] = (lat_df[lat_df.name == int(site['name'])].lat).tolist()[0]
                forage.execute(forage_args)
            # collect biomass at 2015 measurement date for each ecolclass

        if len(graz_filter) == 1:
            shutil.copyfile(os.path.join(century_dir, 'default_graz.100'),
                            def_graz_file)
            os.remove(os.path.join(century_dir, 'default_graz.100'))

    def collect_biomass():
        """summarize biomass at each ecolclass within each property"""

        sum_dict = {'site': [], 'ecolclass': [], 'green_biomass': [],
                    'total_biomass': []}
        site_list = generate_inputs(match_csv, 2015, 'total')
        for site in site_list:
            if site['name'] == '12':
                continue  # skip Lombala
            match_year = 2015
            mo_float = (site['date'] - 2015) * 12.0
            if mo_float - int(mo_float) > 0.5:
                match_month = int(mo_float) + 1
            else:
                match_month = int(mo_float)
            for ecol_class in ['livestock', 'integrated', 'wildlife']:
                out_dir_site = os.path.join(empir_outdir,
                                            '{}_{}'.format(site['name'],
                                                           ecol_class))
                emp_res = pd.read_csv(os.path.join(out_dir_site,
                                                   'summary_results.csv'))
                emp_subs = emp_res.loc[(emp_res.year == match_year) &
                                       (emp_res.month == match_month)]
                if emp_subs.shape[0] != 1:
                    import pdb; pdb.set_trace()
                gre_biomass = emp_subs.iloc[0]['{}_green_kgha'.format(site['name'])] * 0.1
                tot_biomass = (emp_subs.iloc[0]['{}_dead_kgha'.format(site['name'])] +
                               emp_subs.iloc[0]['{}_green_kgha'.format(site['name'])]) * 0.1
                sum_dict['site'].append(site['name'])
                sum_dict['ecolclass'].append(ecol_class)
                sum_dict['green_biomass'].append(gre_biomass)
                sum_dict['total_biomass'].append(tot_biomass)
        sum_df = pd.DataFrame(sum_dict)
        sum_df.to_csv(save_as)
    collect_biomass()

def regional_scenarios():
    """Run scenario analysis for regional properties."""

    # back_calc_regional_avg()
    # run forward from 2014 measurement with empirical numbers (all inputs
    # averaged across properties, including densities of different animal types)
    forage_args = default_forage_args()
    forage_args['start_month'] = 8
    forage_args['num_months'] = 12
    forage_args['latitude'] = 0.324
    forage_args[u'user_define_protein'] = 1
    forage_args['input_dir'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios"
    forage_args['grass_csv'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\regional_grass_avg.csv"
    for ecol_class in ['livestock', 'integrated', 'wildlife']:
        forage_args['herbivore_csv'] = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\regional_scenarios\herbivores_regional_scenarios_{}.csv".format(ecol_class)
        outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_densities\{}".format(ecol_class)
        forage_args['outdir'] = outdir
        forage.execute(forage_args)

def regional_scenarios_by_property():
    """Workflow to run scenario analysis for each property separately"""

    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\regional_PDM_summary.csv"
    outer_outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_forward_from_2014_density_mult"
    comp_list = []
    for density_mult in [1, 1.2, 1.5, 2, 5, 7, 10]:
        empir_outdir = os.path.join(outer_outdir,
                                    '{0:.2f}'.format(density_mult))
        empirical_densities_forward(match_csv, empir_outdir, density_mult)
        comp_df = compare_biomass(match_csv, empir_outdir, density_mult)
        comp_list.append(comp_df)
        erase_intermediate_files(empir_outdir)
    sum_df = pd.concat(comp_list)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\biomass_comparison.csv"
    sum_df.to_csv(save_as)


def scenario_workflow():
    """Workflow that was under 'main' when running scenarios."""
    # regional_scenarios()
    # regional_scenarios_by_property()
    # run_preset_densities()
    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\regional_PDM_summary.csv"
    outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_densities_within_property"
    save_as = os.path.join(outdir, "biomass_summary.csv")
    scenario_mean_ecolclass_by_property(match_csv, outdir, save_as)


if __name__ == "__main__":
    # combine_marg_df()
    # max_viable_density_rainfall_perturbations()
    Laikipia_uniform_density()
