"""Launch Century for multiple sites, to obtain starting conditions for RPM."""
import os
import time
import re
import shutil
from subprocess import Popen

from osgeo import gdal
import pandas

import pygeoprocessing

# directory containing the Century executable, century_46.exe, and necessary
#   supporting files, such as graz.100 and crop.100
_CENTURY_DIRECTORY = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014"

# Century fix file basename. This must reside in the input directory
_FIX_FILE = 'drygfix.100'

# local directory containing Century output variable definitions`
_DROPBOX_DIR = "C:/Users/ginge/Dropbox/NatCap_backup"

# nodata value for initial state variable rasters
_SV_NODATA = -1.0

def write_century_bat(
        century_bat, schedule, output, fix_file, outvars, extend=None):
    """Write a Windows batch file to launch the Century executable.

    Parameters:
        century_bat (string): filename, the batch file will be saved here
        schedule (string): basename of Century schdule file, a text file with
            the extension '.sch'
        output (string): basename of output files that should be written,
            optionally including file extension
        fix_file (string): basename of Century fix file to use in simulation,
            e.g. 'drytrpfix.100'
        outvars (string): basename of text file containing output variables
            that should be collected from the Century results
        extend (bool, default None): is the Century simulation launched by this
            batch file an "extend" simulation?

    Side effects:
        Saves a Windows batch file named `century_bat` in _CENTURY_DIRECTORY

    Returns:
        None

    """
    if schedule[-4:] == '.sch':
        schedule = schedule[:-4]
    if output[-4:] == '.lis':
        output = output[:-4]

    with open(os.path.join(_CENTURY_DIRECTORY, century_bat), 'w') as file:
        file.write('copy ' + fix_file + ' fix.100\n')

        if extend:
            file.write(
                'century_46 -s ' + schedule + ' -n ' + output + ' -e ' +
                extend + ' > ' + output + '_log.txt\n')
        else:
            file.write(
                'century_46 -s ' + schedule + ' -n ' + output + ' > ' +
                output + '_log.txt\n')
        file.write(
            'list100_46 ' + output + ' ' + output + ' ' + outvars + '\n\n')
        file.write('erase fix.100\n')


def get_site_weather_files(schedule_file):
    """Retrieve basenames of site and weather files from schedule file.

    A Century schedule file (*.sch file) must contain a reference to the name
    of the site file (*.100 file) used for the simulation, and a reference to
    the name of the weather file (*.wth file) if one is used. Retrieve these
    file names from the schedule file.

    Parameters:
        schedule_file (string): path to Century schedule file, with the '.sch'
            file extension

    Raises:
        ValueError if schedule file contains references to more than one site
            file, or more than one unique weather file

    Returns:
        a tuple of strings indicating the basename of the site file and the
            weather file, in that order. If one of these is not specified in
            the schedule file, the corresponding entry in the tuple is 'NA'

    """
    s_name = 'NA'
    w_name = 'NA'
    with open(schedule_file, 'r') as read_file:
        for line in read_file:
            if 'Site file name' in line:
                if s_name != 'NA':
                    er = "Error: two site files found in schedule file"
                    raise ValueError(er)
                else:
                    s_name = re.search('(.+?).100', line).group(1) + '.100'
            if '.wth' in line:
                if w_name != 'NA':
                    second_w_name = re.search(
                        '(.+?).wth', line).group(1) + '.wth'
                    if second_w_name != w_name:
                        er = "Error: two weather files found in schedule file"
                        raise ValueError(er)
                else:
                    w_name = re.search('(.+?).wth', line).group(1) + '.wth'
    return s_name, w_name


def launch_century_subprocess(bat_file):
    """Launch Century executable and check that it completed successfully.

    Run the Century executable by launching a Windows batch file. Check the
    Century log and ensure that it indicates success.

    Parameters:
        bat_file (string): path to Windows batch file

    Raises:
        ValueError if the Century log indicates that Century did not complete
            successfully

    Returns:
        None

    """
    p = Popen(["cmd.exe", "/c " + bat_file], cwd=_CENTURY_DIRECTORY)
    stdout, stderr = p.communicate()
    p.wait()
    log_file = bat_file[:-4] + "_log.txt"
    success = 0
    error = []
    num_tries = 3
    tries = 0
    while tries < num_tries:
        with open(log_file, 'r') as file:
            for line in file:
                if 'Execution success.' in line:
                    success = 1
                    return
        if not success:
            with open(log_file, 'r') as file:
                error = [line.strip() for line in file]
                if len(error) == 0:
                    error = "CENTURY log file is empty"
                time.sleep(1.0)
                tries = tries + 1
    if error == ['', 'Model is running...']:  # special case?
        return
    raise ValueError(error)


def launch_sites(site_csv, shp_id_field, input_dir, outer_outdir):
    """Launch Century executable for a series of sites.

    For each of a series of sites expressed as rows in a csv table, fetch
    inputs to run Century, write a Windows batch table, run Century, and move
    the resulting outputs into a new folder for the site.

    Parameters:
        site_csv (string): path to a table containing coordinates labels
            for a series of sites.  Must contain a column, shp_id_field, which
            is a site label that matches basename of inputs in `input_dir` that
            may be used to run Century
        shp_id_field (string): site label, included as a field in `site_csv`
            and used as basename of Century input files
        input_dir (string): path to a directory containing Century input files
            for each site in `site_csv`
        outer_outdir (string): path to a directory where Century results should
            be saved. A new interior folder will be created here for each site

    Side effects:
        Creates inner directories, one per row in `site_csv`, in `outer_outdir`
        Launches the Century executable and moves results to new inner folders

    Returns:
        None

    """
    # assume fix file is in the input directory, copy it to Century
    #   directory
    shutil.copyfile(
        os.path.join(input_dir, _FIX_FILE),
        os.path.join(_CENTURY_DIRECTORY, _FIX_FILE))

    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    for site in site_list:
        site_id = site[shp_id_field]
        outdir = os.path.join(outer_outdir, '{}'.format(site_id))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # write Century batch file for spin-up simulation
        hist_bat_bn = '{}_hist.bat'.format(site_id)
        hist_bat = os.path.join(_CENTURY_DIRECTORY, hist_bat_bn)
        hist_schedule = '{}_hist.sch'.format(site_id)
        hist_output = '{}_hist'.format(site_id)
        write_century_bat(
            hist_bat, hist_schedule, hist_output, _FIX_FILE, 'outvars.txt')

        # write Century batch file for extend simulation
        extend_bat_bn = '{}.bat'.format(site_id)
        extend_bat = os.path.join(_CENTURY_DIRECTORY, extend_bat_bn)
        schedule = '{}.sch'.format(site_id)
        output = '{}.lis'.format(site_id)
        extend = '{}_hist'.format(site_id)
        write_century_bat(
            extend_bat, schedule, output, _FIX_FILE, 'outvars.txt', extend)

        # move Century input files to Century dir
        input_files = []
        input_files.append('{}.sch'.format(site_id))
        input_files.append('{}_hist.sch'.format(site_id))
        site_file, weather_file = get_site_weather_files(
            os.path.join(input_dir, '{}.sch'.format(site_id)))
        input_files.append(site_file)
        if weather_file != 'NA':
            input_files.append(weather_file)
        for input_basename in input_files:
            shutil.copyfile(
                os.path.join(input_dir, input_basename),
                os.path.join(_CENTURY_DIRECTORY, input_basename))

        # run Century
        try:
            launch_century_subprocess(hist_bat)
            launch_century_subprocess(extend_bat)
        finally:
            # move Century outputs to results folder
            output_files = [
                '{}_hist_log.txt'.format(site_id), '{}_hist.lis'.format(site_id),
                '{}_hist.bin'.format(site_id),
                '{}_log.txt'.format(site_id), '{}.lis'.format(site_id),
                '{}.bin'.format(site_id)]
            for output_basename in output_files:
                if os.path.exists(
                        os.path.join(_CENTURY_DIRECTORY, output_basename)):
                    shutil.move(
                        os.path.join(_CENTURY_DIRECTORY, output_basename),
                        os.path.join(outdir, output_basename))

            # remove batch files and input files from Century directory
            files_to_remove = input_files + [hist_bat_bn, extend_bat_bn]
            for filename in files_to_remove:
                if os.path.exists(os.path.join(_CENTURY_DIRECTORY, filename)):
                    os.remove(os.path.join(_CENTURY_DIRECTORY, filename))
    # clean up fix file
    os.remove(os.path.join(_CENTURY_DIRECTORY, _FIX_FILE))


def convert_to_century_date(year, month):
    """Convert year and month to Century's representation of dates."""
    return float('%.2f' % (year + month / 12.))


def century_to_rpm(century_label):
    """Convert Century variable name to rangeland production model name."""
    rp = re.sub(r"\(", "_", century_label)
    rp = re.sub(r",", "_", rp)
    rp = re.sub(r"\)", "", rp)
    return rp


def century_outputs_to_rpm_initial_rasters(
        site_csv, outer_outdir, year, month, site_index_path,
        initial_conditions_dir):
    """Generate initial conditions rasters for RPM from raw Century outputs.

    Take outputs from a series of Century runs and convert them to initial
    conditions rasters, one per state variable, for the Rangeland Production
    Model.

    Parameters:
        site_csv (string): path to a table containing coordinates labels
            for a series of sites.  Must contain a column, 'site_id', that
            identifies the Century outputs pertaining to each site; and another
            column, 'site_id_num', that is a unique integer code for each site
        outer_outdir (string): path to a directory containing Century output
            files. It is expected that this directory contains a separate
            folder of outputs for each site
        year (integer): year of the date from which to draw initial values
        month (integer): month of the date from which to draw initial values
        site_index_path (string): path to raster that indexes sites spatially,
            indicating which set of Century outputs should apply at each pixel
            in the raster. E.g., this raster could contain Thiessen polygons
            corresponding to a set of points where Century has been run
        initial_conditions_dir (string): path to directory where initial
            conditions rasters should be written

    Side effects:
        creates or modifies rasters in `initial_conditions_dir`, one per state
            variable required to initialize RPM

    Returns:
        None

    """
    if not os.path.exists(initial_conditions_dir):
        os.makedirs(initial_conditions_dir)
    time = convert_to_century_date(year, month)

    # Century output variables
    outvar_csv = os.path.join(
        _DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
    outvar_df = pandas.read_csv(outvar_csv)
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    outvar_df.sort_values(by=['outvar'], inplace=True)

    site_df_list = []
    pft_df_list = []
    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    for site in site_list:
        site_id = site['site_id']
        site_id_num = site['site_id_num']
        century_output_file = os.path.join(
            outer_outdir, '{}'.format(site_id), '{}.lis'.format(site_id))
        test_output_list = outvar_df[
            outvar_df.Property_of == 'PFT'].outvar.tolist()
        cent_df = pandas.read_fwf(century_output_file, skiprows=[1])
        # mistakes in Century writing results
        if 'minerl(10,1' in cent_df.columns.values:
            cent_df.rename(
                index=str, columns={'minerl(10,1': 'minerl(10,1)'},
                inplace=True)
        if 'minerl(10,2' in cent_df.columns.values:
            cent_df.rename(
                index=str, columns={'minerl(10,2': 'minerl(10,2)'},
                inplace=True)
        try:
            fwf_correct = cent_df[test_output_list]
        except KeyError:
            # try again, specifying widths explicitly
            widths = [16] * 79
            cent_df = pandas.read_fwf(
                century_output_file, skiprows=[1], widths=widths)
            # mistakes in Century writing results
            if 'minerl(10,1' in cent_df.columns.values:
                cent_df.rename(
                    index=str, columns={'minerl(10,1': 'minerl(10,1)'},
                    inplace=True)
            if 'minerl(10,2' in cent_df.columns.values:
                cent_df.rename(
                    index=str, columns={'minerl(10,2': 'minerl(10,2)'},
                    inplace=True)
        df_subset = cent_df[(cent_df.time == time)]
        df_subset = df_subset.drop_duplicates('time')
        for sbstr in ['PFT', 'site']:
            output_list = outvar_df[
                outvar_df.Property_of == sbstr].outvar.tolist()
            try:
                outputs = df_subset[output_list]
            except KeyError:
                import pdb; pdb.set_trace()  # WTF
            outputs = outputs.loc[:, ~outputs.columns.duplicated()]
            col_rename_dict = {
                c: century_to_rpm(c) for c in outputs.columns.values}
            outputs.rename(index=int, columns=col_rename_dict, inplace=True)
            outputs[sbstr] = site_id_num
            if sbstr == 'site':
                site_df_list.append(outputs)
            if sbstr == 'PFT':
                pft_df_list.append(outputs)
    site_initial_df = pandas.concat(site_df_list)
    site_initial_df.set_index('site', inplace=True)
    siteid_to_initial = site_initial_df.to_dict(orient='index')
    site_sv_list = outvar_df[outvar_df.Property_of == 'site'].outvar.tolist()
    rpm_site_sv_list = [century_to_rpm(c) for c in site_sv_list]
    for site_sv in rpm_site_sv_list:
        site_to_val = dict(
            [(site_code, float(table[site_sv])) for (site_code, table) in
            siteid_to_initial.items()])
        target_path = os.path.join(
            initial_conditions_dir, '{}.tif'.format(site_sv))
        pygeoprocessing.reclassify_raster(
            (site_index_path, 1), site_to_val, target_path,
            gdal.GDT_Float32, _SV_NODATA)

    pft_initial_df = pandas.concat(pft_df_list)
    pft_initial_df.set_index('PFT', inplace=True)
    pft_to_initial = pft_initial_df.to_dict(orient='index')
    pft_sv_list = outvar_df[outvar_df.Property_of == 'PFT'].outvar.tolist()
    rpm_pft_sv_list = [century_to_rpm(c) for c in pft_sv_list]
    for pft_sv in rpm_pft_sv_list:
        site_to_pftval = dict(
            [(site_code, float(table[pft_sv])) for (site_code, table) in
            pft_to_initial.items()])
        target_path = os.path.join(
            initial_conditions_dir, '{}_{}.tif'.format(pft_sv, 1))
        pygeoprocessing.reclassify_raster(
            (site_index_path, 1), site_to_pftval, target_path,
            gdal.GDT_Float32, _SV_NODATA)


def summarize_century_biomass(
        site_csv, shp_id_field, outer_outdir, start_time, end_time, save_as):
    """Collect outputs from Century results at a series of sites.

    Make a table of simulated biomass, including live and standing dead, for
    the period `start_time` to `end_time` from raw Century outputs.

    Parameters:
        site_csv (string): path to a table containing coordinates and labels
            for a series of sites.  Must contain a column, 'site_id', that
            identifies the Century outputs pertaining to each site
        shp_id_field (string): site label, included as a field in `site_csv`
            and used as basename of Century input files
        outer_outdir (string): path to a directory containing Century output
            files. It is expected that this directory contains a separate
            folder of outputs for each site
        start_time (float): starting date for which outputs should be collected
        end_time (float): ending date for which outputs should be collected
        save_as (string): path to location on disk where summarized biomass
            should be saved

    Side effects:
        creates or modifies a csv file at `save_as`

    Returns:
        None

    """
    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    df_list = []
    for site in site_list:
        site_id = site[shp_id_field]
        output_file = os.path.join(
            outer_outdir, '{}'.format(site_id), '{}.lis'.format(site_id))
        cent_df = pandas.read_fwf(output_file, skiprows=[1])
        cent_df = cent_df[
            (cent_df.time >= start_time) & (cent_df.time <= end_time)]
        cent_df = cent_df[['time', 'aglivc', 'stdedc']]
        cent_df['live_biomass'] = cent_df.aglivc * 2.5
        cent_df['dead_biomass'] = cent_df.stdedc * 2.5
        cent_df['total_biomass'] = cent_df.live_biomass + cent_df.dead_biomass
        cent_df.drop_duplicates(inplace=True)
        cent_df['site_id'] = site_id
        cent_df.set_index('time', inplace=True)
        df_list.append(cent_df)
    sum_df = pandas.concat(df_list)
    sum_df.to_csv(save_as)


def wcs_monitoring_points_workflow():
    """Run Century to get initialization rasters for WCS monitoring area."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/soil/monitoring_points_soil_isric_250m.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/chirps_prec_back_calc"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/monitoring_sites/chirps_prec_back_calc_2.13.20"
    launch_sites(site_csv, input_dir, outer_outdir)
    year = 2016
    month = 8
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/site_idx_CBM_SCP_voronoi_polygon.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/initial_conditions"
    century_outputs_to_rpm_initial_rasters(
        site_csv, outer_outdir, year, month, site_index_path,
        initial_conditions_dir)
    start_time = 2015.08
    end_time = 2018.00
    save_as = os.path.join(outer_outdir, 'biomass_summary.csv')
    summarize_century_biomass(
        site_csv, outer_outdir, start_time, end_time, save_as)


def julian_ahlborn_sites_workflow():
    """Run Century to get initialization rasters for Julian Ahlborn's sites."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_outputs"
    # launch_sites(site_csv, 'site', input_dir, outer_outdir)
    start_time = 2014.08
    end_time = 2015.00
    save_as = os.path.join(outer_outdir, 'biomass_summary.csv')
    summarize_century_biomass(
        site_csv, 'site', outer_outdir, start_time, end_time, save_as)


def main():
    """Program entry point."""
    julian_ahlborn_sites_workflow()


if __name__ == "__main__":
    main()
