# run forage model on regional properties

import os
import sys
import re
import shutil
import math
from tempfile import mkstemp
import pandas as pd
import numpy as np
import back_calculate_management as backcalc
sys.path.append(r'C:\Users\Ginger\Documents\Python\rangeland_production')
import forage_century_link_utils as cent
import forage

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
            'mgmt_threshold': 0.1,
            'century_dir': 'C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'template_level': 'GH',
            'fix_file': 'drytrpfi.100',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'supp_csv': "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Rubanza_et_al_2005_supp.csv",
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
        print "the following sites failed:"
        print failed

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
        
if __name__ == "__main__":
    # regional_scenarios()
    # regional_scenarios_by_property()
    # run_preset_densities()
    match_csv = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\From_Sharon\Processed_by_Ginger\regional_PDM_summary.csv"
    outdir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_results\regional_scenarios\empirical_densities_within_property"
    save_as = os.path.join(outdir, "biomass_summary.csv")
    scenario_mean_ecolclass_by_property(match_csv, outdir, save_as)
