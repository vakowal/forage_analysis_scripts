"""Back-calculate management routine.

The back-calculate management routine estimates historical grazing pressure
prior to an empirical biomass collection date by iteratively modifying the
historical grazing schedule until predicted biomass modeled by Century matches
the empirical biomass, or until a maximum number of iterations are reached.

"""
import math
import csv
import os
import sys
import shutil

import pandas as pd
import re
sys.path.append(
 'C:/Users/ginge/Documents/Python/rangeland_production')
import forage_century_link_utils as cent


def back_calculate_management(
        site, input_dir, century_dir, out_dir, fix_file, n_months, vary,
        live_or_total, threshold, max_iterations, template_level):
    """Back-calculate grazing history at a site to match empirical biomass.

    This function uses Century to predict biomass at the site on the empirical
    measurement date, and iteratively increases or decreases herbivore grazing
    pressure prior to the empirical measurement date until modeled biomass on
    the empirical measurement date matches empirical biomass, or until a
    certain number of iterations are completed.

    Parameters:
        site (dict): dictionary describing the empirical measurement that the
                routine should match. This dictionary must contain the
                following keys:
            - biomass: empirical biomass, in grams per square meter
            - date: empirical measurement date, in decimal format with two
              decimal places. For example, 2012.00 designates December 2011;
              2012.08 designates January 2012
            - name: site identifier. This must be matched to inputs to the beta
              rangeland model for the site, such as the schedule and <site>.100
              files.
        input_dir (string): path to directory containing inputs to the beta
            rangeland model for each site, for example schedule and <site>.100
            files
        century_dir (string): path to directory containing Century executable
            and supporting files, such as graz.100
        out_dir (string): path to directory where outputs from the routine
            should be written
        fix_file (string): basename of the Century fix file that should be used
            for the Century model run. This file must be located in
            `century_dir`.
        n_months (int): number of months prior to the empirical sampling date
            that the back-calculate management routine should potentially
            modify
        vary (string): how should the grazing history be changed? one of
            'intensity', 'schedule', or 'both'
        live_or_total (string): match live, or live+standing dead biomass?
            one of 'live' or 'total'
        threshold (float): maximum difference between simulated and empirical
            biomass, in grams per square m, to consider the match successful
        max_iterations (int): maximum number of times to run Century with
            modified grazing schedule before abandoning the attempt
        template_level (string): an existing grazing code in graz.100 to
            use as the basis for modifed grazing events, giving the set of
            grazing-related Century parameters (e.g., gfcret, gret(1), grzeff)

    Side effects:
        Creates output files in `out_dir`

    Returns:
        None

    """
    cent.set_century_directory(century_dir)
    empirical_biomass = site['biomass']
    empirical_date = site['date']
    schedule = site['name'] + '.sch'
    schedule_file = os.path.join(input_dir, schedule)
    site_file, weather_file = cent.get_site_weather_files(
        schedule_file, input_dir)
    graz_file = os.path.join(century_dir, "graz.100")
    output = site['name']

    # check that last block in schedule file includes >= n_months before
    # empirical_date
    cent.check_schedule(schedule_file, n_months, empirical_date)

    # write CENTURY batch file for spin-up simulation
    hist_bat = os.path.join(input_dir, (site['name'] + '_hist.bat'))
    hist_schedule = site['name'] + '_hist.sch'
    hist_output = site['name'] + '_hist'
    cent.write_century_bat(
        input_dir, hist_bat, hist_schedule, hist_output, fix_file,
        'outvars.txt')
    # write CENTURY bat for extend simulation
    extend_bat = os.path.join(input_dir, (site['name'] + '.bat'))
    extend = site['name'] + '_hist'
    output_lis = output + '.lis'
    cent.write_century_bat(
        input_dir, extend_bat, schedule, output_lis, fix_file, 'outvars.txt',
        extend)
    # make a copy of the original graz params and schedule file
    shutil.copyfile(graz_file, os.path.join(century_dir, 'graz_orig.100'))
    shutil.copyfile(schedule_file, os.path.join(input_dir, 'sch_orig.sch'))

    spin_up_outputs = [
        site['name'] + '_hist_log.txt', site['name'] + '_hist.lis']
    century_outputs = [output + '_log.txt', output + '.lis', output + '.bin']
    # move CENTURY run files to CENTURY dir
    e_schedule = os.path.join(input_dir, site['name'] + '.sch')
    h_schedule = os.path.join(input_dir, site['name'] + '_hist.sch')
    file_list = [
        hist_bat, extend_bat, e_schedule, h_schedule, site_file, weather_file]
    if weather_file == 'NA':
        file_list.remove(weather_file)
    for file in file_list:
        shutil.copyfile(file, os.path.join(
            century_dir, os.path.basename(file)))
    run_schedule = os.path.join(century_dir, os.path.basename(e_schedule))
    os.remove(hist_bat)
    os.remove(extend_bat)
    # run CENTURY for spin-up
    hist_bat_run = os.path.join(century_dir, (site['name'] + '_hist.bat'))
    century_bat_run = os.path.join(century_dir, (site['name'] + '.bat'))
    cent.launch_CENTURY_subprocess(hist_bat_run)

    # save copies of CENTURY outputs, but remove from CENTURY dir
    intermediate_dir = os.path.join(out_dir, 'CENTURY_outputs_spin_up')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    for file in spin_up_outputs:
        shutil.copyfile(
            os.path.join(century_dir, file),
            os.path.join(intermediate_dir, file))
        os.remove(os.path.join(century_dir, file))
    # start summary csv
    summary_csv = os.path.join(
        out_dir, 'modify_management_summary_' + site['name'] + '.csv')
    try:
        with open(summary_csv, 'wb') as summary_file:
            writer = csv.writer(summary_file, delimiter=',')
            row = ['Iteration', 'Empirical_biomass', 'Simulated_biomass']
            writer.writerow(row)
            for iter in xrange(max_iterations):
                row = [iter]
                row.append(empirical_biomass)

                # call CENTURY from the batch file
                cent.launch_CENTURY_subprocess(century_bat_run)

                intermediate_dir = os.path.join(
                    out_dir, 'CENTURY_outputs_iteration%d' % iter)
                if not os.path.exists(intermediate_dir):
                    os.makedirs(intermediate_dir)
                for file in century_outputs:
                    shutil.copyfile(
                        os.path.join(century_dir, file),
                        os.path.join(intermediate_dir, file))
                    os.remove(os.path.join(century_dir, file))

                # get simulated biomass
                output_file = os.path.join(intermediate_dir, output_lis)
                biomass_df = cent.read_CENTURY_outputs(
                    output_file, math.floor(empirical_date) - 1,
                    math.ceil(empirical_date) + 1)
                biomass_df.drop_duplicates(inplace=True)
                if live_or_total == 'live':
                    simulated_biomass = biomass_df.loc[
                        empirical_date, 'aglivc']
                elif live_or_total == 'total':
                    simulated_biomass = biomass_df.loc[
                        empirical_date, 'total']
                else:
                    er = """Biomass output from CENTURY must be specified as
                            'live' or 'total'"""
                    raise ValueError(er)
                row.append(simulated_biomass)
                writer.writerow(row)

                # check difference between simulated and empirical biomass
                if float(abs(
                        simulated_biomass - empirical_biomass)) <= threshold:
                    print("Biomass target reached!")
                    break  # success!
                else:
                    # these are the guts of the routine, where either the
                    # grazing parameters file or the schedule file used in the
                    # CENTURY call are changed
                    diff = simulated_biomass - empirical_biomass
                    increase_intensity = 0
                    if diff > 0:  # simulated biomass greater than empirical
                        increase_intensity = 1
                    if vary == 'intensity':
                        # change % biomass removed in grazing parameters file
                        # TODO the grazing level here should reflect what's in
                        # the schedule file
                        success = cent.modify_intensity(
                            increase_intensity, graz_file, template_level,
                            out_dir, site['name'] + str(iter))
                        if not success:
                            print("""intensity cannot be modified, %d
                                     iterations completed""" % (iter - 1))
                            break
                    elif vary == 'schedule':
                        # add or remove scheduled grazing events
                        target_dict = cent.find_target_month(
                            increase_intensity, run_schedule, empirical_date,
                            n_months)
                        cent.modify_schedule(
                            run_schedule, increase_intensity, target_dict,
                            template_level, out_dir, site['name'] + str(iter))
                    elif vary == 'both':
                        target_dict = cent.find_target_month(
                            increase_intensity, run_schedule, empirical_date,
                            n_months)
                        if target_dict:
                            # there are opportunities to modify the schedule
                            cent.modify_schedule(
                                run_schedule, increase_intensity, target_dict,
                                template_level, out_dir, site['name'] +
                                str(iter))
                        else:
                            # no opportunities to modify the schedule exist
                            success = cent.modify_intensity(
                                increase_intensity, graz_file, template_level,
                                out_dir, site['name'] + str(iter))
                            if not success:
                                print("""intensity cannot be modified, %d
                                         iterations completed""" % (iter - 1))
                                break
                if iter == (max_iterations - 1):
                    print('maximum iterations performed, target not reached')
    finally:
        # replace modified grazing parameters and schedule files with original
        # files
        os.remove(graz_file)
        shutil.copyfile(os.path.join(century_dir, 'graz_orig.100'), graz_file)
        os.remove(os.path.join(century_dir, 'graz_orig.100'))
        os.remove(schedule_file)
        shutil.copyfile(
            os.path.join(input_dir, 'sch_orig.sch'), schedule_file)
        files_to_remove = [
            os.path.join(century_dir, os.path.basename(f)) for f in file_list]
        for file in files_to_remove:
            os.remove(file)
        os.remove(os.path.join(input_dir, 'sch_orig.sch'))
        os.remove(os.path.join(century_dir, site['name'] + '_hist.bin'))


def summarize_calc_schedules(
        site_list, n_months, input_dir, century_dir, out_dir, raw_file,
        summary_file):
    """Summarize grazing schedule estimated by back-calculate management.

    Access the ending schedules for sites where grazing pressure was estimated
    by the back-calculate management routine.  Summarize the biomass removed
    by herbivores according to the estimated schedules.

    Parameters:
        site_list (dict): a dictionary containing information about the sites;
            this must be the same dictionary that was supplied to the back-
            calculate management routine
        n_months (int): number of months that the back-calculate management
            routine was allowed to potentially modify
        input_dir (string): path to directory containing inputs to the beta
            rangeland model for each site, for example schedule and <site>.100
            files
        century_dir (string): path to directory containing Century executable
            and supporting files, such as graz.100
        out_dir (string): path to directory where the results of the back-
            calculate management routine were written.
        raw_file (string): path to csv file where results table should be
            written, including biomass removed in each month at each site
        summary_file (string): path to csv file where summary results table
            should be written, including average biomass removed at each site
            across months

    Side effects:
        writes a csv table to `raw_file` and `summary_file`

    Returns:
        None

    """
    df_list = list()
    for site in site_list:
        site_name = site['name']
        empirical_date = site['date']
        site_dir = os.path.join(out_dir, '{}'.format(site_name))
        result_csv = os.path.join(
            site_dir, 'modify_management_summary_{}.csv'.format(site_name))
        res_df = pd.read_csv(result_csv)
        sch_files = [f for f in os.listdir(site_dir) if f.endswith('.sch')]
        sch_iter_list = [
            int(re.search('{}_{}(.+?).sch'.format(
                site_name, site_name), f).group(1)) for f in sch_files]
        if len(sch_iter_list) == 0:  # no schedule modification was needed
            final_sch = os.path.join(input_dir, '{}.sch'.format(site_name))
        else:
            final_sch_iter = max(sch_iter_list)
            final_sch = os.path.join(
                site_dir, '{}_{}{}.sch'.format(
                    site_name, site_name, final_sch_iter))

        # read schedule file, collect months where grazing was scheduled
        schedule_df = cent.read_block_schedule(final_sch)
        for i in range(0, schedule_df.shape[0]):
            start_year = schedule_df.loc[i, 'block_start_year']
            last_year = schedule_df.loc[i, 'block_end_year']
            if empirical_date > start_year and empirical_date <= last_year:
                break
        relative_empirical_year = int(
            math.floor(empirical_date) - start_year + 1)
        empirical_month = int(
            round((empirical_date - float(math.floor(empirical_date))) * 12))
        if empirical_month == 0:
            empirical_month = 12
            relative_empirical_year = relative_empirical_year - 1
        first_rel_month, first_rel_year = cent.find_first_month_and_year(
            n_months, empirical_month, relative_empirical_year)
        first_abs_year = first_rel_year + start_year - 1
        # find months where grazing took place prior to empirical date
        graz_schedule = cent.read_graz_level(final_sch)
        block = graz_schedule.loc[
            (graz_schedule["block_end_year"] == last_year),
            ['relative_year', 'month', 'grazing_level', 'year']]
        empirical_year = block.loc[
            (block['relative_year'] == relative_empirical_year) &
            (block['month'] <= empirical_month), ]
        intervening_years = block.loc[
            (block['relative_year'] < relative_empirical_year) &
            (block['relative_year'] > first_rel_year), ]
        first_year = block.loc[
            (block['relative_year'] == first_rel_year) &
            (block['month'] >= first_rel_month), ]
        history = pd.concat([first_year, intervening_years, empirical_year])
        if len(history) > 0:
            # collect % biomass removed for these months
            grz_files = [
                f for f in os.listdir(site_dir) if f.startswith('graz')]
            if len(grz_files) > 0:
                grz_iter_list = [
                    int(re.search(
                        'graz_{}(.+?).100'.format(site_name), f).group(1)) for
                    f in grz_files]
                final_iter = max(grz_iter_list)
                final_grz = os.path.join(
                    site_dir, 'graz_{}{}.100'.format(site_name, final_iter))
            else:
                final_grz = os.path.join(century_dir, 'graz.100')
            grz_levels = history.grazing_level.unique()
            grz_level_list = [{'label': level} for level in grz_levels]
            for grz in grz_level_list:
                flgrem, fdgrem = retrieve_grazing_params(
                    final_grz, grz['label'])
                grz['flgrem'] = flgrem
                grz['fdgrem'] = fdgrem
        # fill in history with months where no grazing took place
        history = cent.fill_schedule(
            history, first_rel_year, first_rel_month, relative_empirical_year,
            empirical_month)
        history['year'] = history['relative_year'] + start_year - 1
        history['perc_live_removed'] = 0.
        history['perc_dead_removed'] = 0.
        if len(grz_level_list) > 0:
            for grz in grz_level_list:
                history.ix[
                    history.grazing_level == grz['label'],
                    'perc_live_removed'] = grz['flgrem']
                history.ix[
                    history.grazing_level == grz['label'],
                    'perc_dead_removed'] = grz['fdgrem']
        # collect biomass for these months
        history['date'] = history.year + history.month / 12.0
        history = history.round({'date': 2})
        final_sim = int(res_df.iloc[len(res_df) - 1].Iteration)
        output_file = os.path.join(
            site_dir, 'CENTURY_outputs_iteration{}'.format(final_sim),
            '{}.lis'.format(site_name))
        biomass_df = cent.read_CENTURY_outputs(
            output_file, first_abs_year, math.floor(empirical_date))
        biomass_df['time'] = biomass_df.index
        sum_df = history.merge(
            biomass_df, left_on='date', right_on='time', how='inner')
        sum_df['site'] = site_name
        df_list.append(sum_df)
    summary_df = pd.concat(df_list)
    summary_df['live_rem'] = summary_df.aglivc * summary_df.perc_live_removed
    summary_df['dead_rem'] = summary_df.stdedc * summary_df.perc_dead_removed
    summary_df['total_rem'] = summary_df.live_rem + summary_df.dead_rem
    rem_means = summary_df.groupby(by='site')[
        ('total_rem', 'live_rem', 'dead_rem')].mean()
    rem_means.to_csv(summary_file)
    summary_df.to_csv(raw_file)


def retrieve_grazing_params(grz_file, grz_level):
    """Get parameters 'flgrem' and 'fdgrem' for a particular grazing level.

    Retrieve the parameters 'flgrem' (fraction of live biomass removed) and
    'fdgrem' (fraction of dead biomass removed) from the given grazing level
    in the given grazing file.

    Parameters:
        grz_file (string): path to graz.100 file containing Century parameters
            related to grazing
        grz_level (string): grazing level identifying a set of grazing
            parameters in the graz.100 file, e.g., 'GH', 'GL', 'GM'

    Returns:
        flgrem, fdgrem as a tuple

    """
    with open(grz_file, 'rb') as grz:
        for line in grz:
            if '{}  '.format(grz_level) in line:
                line = grz.next()
                if 'FLGREM' in line:
                    flgrem = float(line[:8].strip())
                    line = grz.next()
                if 'FDGREM' in line:
                    fdgrem = float(line[:8].strip())
                else:
                    er = "Error: FLGREM expected"
                    raise Exception(er)
    return flgrem, fdgrem
