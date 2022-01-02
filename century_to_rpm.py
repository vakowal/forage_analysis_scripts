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


def launch_sites(
        site_csv, shp_id_field, input_dir, outer_outdir, fix_file=None):
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
        fix_file (string): basename of Century fix file. If none, the fix file
            used for Century simulations has the global value _FIX_FILE

    Side effects:
        Creates inner directories, one per row in `site_csv`, in `outer_outdir`
        Launches the Century executable and moves results to new inner folders

    Returns:
        None

    """
    if fix_file:
        global _FIX_FILE
        _FIX_FILE = fix_file
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
            hist_bat, hist_schedule, hist_output, _FIX_FILE, 'outvars_old.txt')  # TODO replace 'outvars.txt')

        # write Century batch file for extend simulation
        extend_bat_bn = '{}.bat'.format(site_id)
        extend_bat = os.path.join(_CENTURY_DIRECTORY, extend_bat_bn)
        schedule = '{}.sch'.format(site_id)
        output = '{}.lis'.format(site_id)
        extend = '{}_hist'.format(site_id)
        write_century_bat(
            extend_bat, schedule, output, _FIX_FILE, 'outvars_old.txt', extend)  # TODO replace 'outvars.txt')

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


def century_npp_to_raster(
        site_csv, shp_id_field, outer_outdir, site_index_path, target_path):
    """Make a raster of NPP from Century outputs at gridded points.

    Assume we want to calculate average 'cproda' from month 12 in years 2014,
    2015, and 2016.

    Parameters:
        site_csv (string): path to a table containing coordinates labels
            for a series of sites.  Must contain a column, shp_id_field, which
            is a site label that matches basename of inputs in `input_dir` that
            may be used to run Century
        shp_id_field (string): site label, included as a field in `site_csv`
            and used as basename of Century input files
        outer_outdir (string): path to a directory containing Century output
            files. It is expected that this directory contains a separate
            folder of outputs for each site
        site_index_path (string): path to raster that indexes sites spatially,
            indicating which set of Century outputs should apply at each pixel
            in the raster. E.g., this raster could contain Thiessen polygons
            corresponding to a set of points where Century has been run
        target_path (string): path where npp raster should be written

    """
    site_to_val = {}
    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    for site in site_list:
        site_id = site[shp_id_field]
        raster_map_value = site_id
        century_output_file = os.path.join(
            outer_outdir, '{}'.format(site_id), '{}.lis'.format(site_id))
        cent_df = pandas.read_fwf(century_output_file, skiprows=[1])
        mean_cproda = (cent_df[
            (cent_df.time == 2015.00) |
            (cent_df.time == 2016.00) |
            (cent_df.time == 2017.00)]['cproda']).mean()
        site_to_val[site_id] = mean_cproda
    pygeoprocessing.reclassify_raster(
        (site_index_path, 1), site_to_val, target_path,
        gdal.GDT_Float32, _SV_NODATA)


def century_outputs_to_rpm_initial_rasters(
        site_csv, shp_id_field, outer_outdir, year, month, site_index_path,
        initial_conditions_dir, raster_id_field=None):
    """Generate initial conditions rasters for RPM from raw Century outputs.

    Take outputs from a series of Century runs and convert them to initial
    conditions rasters, one per state variable, for the Rangeland Production
    Model.

    Parameters:
        site_csv (string): path to a table containing coordinates labels
            for a series of sites.  Must contain a column, shp_id_field, which
            is a site label that matches basename of inputs in `input_dir` that
            may be used to run Century
        shp_id_field (string): site label, included as a field in `site_csv`
            and used as basename of Century input files
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
        raster_id_field (integer): field in `site_csv` that corresponds to
            values in `site_index_path`. If this is none, it is assumed that
            this field is the same as `shp_index_field`

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
        site_id = site[shp_id_field]
        if raster_id_field:
            raster_map_value = site[raster_id_field]
        else:
            raster_map_value = site_id
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
            outputs[sbstr] = raster_map_value
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


def century_outputs_to_initial_tables(
        century_output_file, year, month, site_initial_table,
        pft_initial_table):
    """Generate initial value tables from raw Century outputs.

    Take outputs from a single Century run and format them as initial values
    tables for the Rangeland model. The resulting initial values tables will
    represent one site type and one plant functional type, both indexed by '1'.

    Parameters:
        century_output_file (string): path to output file containing Century
            outputs, e.g. the file extension of this file should be '.lis'
        year (integer): year of the date from which to draw initial values
        month (integer): month of the date from which to draw initial values
        site_initial_table (string): path to filename where site initial value
            table should be created
        pft_initial_table (string): path to filename where plant functional
            type initial value table should be created

    Side effects:
        creates or modifies the csv file indicated by `site_initial_table`
        creates or modifies the csv file indicated by `pft_initial_table`

    Returns:
        None

    """
    def century_to_rp(century_label):
        """Convert Century name to rangeland production name."""
        rp = re.sub(r"\(", "_", century_label)
        rp = re.sub(r",", "_", rp)
        rp = re.sub(r"\)", "", rp)
        return rp

    time = convert_to_century_date(year, month)
    outvar_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv"
    outvar_df = pandas.read_csv(outvar_csv)
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    outvar_df.sort_values(by=['outvar'], inplace=True)
    for sbstr in ['PFT', 'site']:
        output_list = outvar_df[
            outvar_df.Property_of == sbstr].outvar.tolist()
        cent_df = pandas.read_fwf(century_output_file, skiprows=[1])
        # mistakes in Century writing results
        if 'minerl(10,1' in cent_df.columns.values:
            cent_df.rename(
                index=str, columns={'minerl(10,1': 'minerl(10,1)'},
                inplace=True)
        if 'minerl(10,2' in cent_df.columns.values:
            cent_df.rename(
                index=str, columns={'minerl(10,2': 'struce(2,2)'},
                inplace=True)
        try:
            fwf_correct = cent_df[output_list]
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
        outputs = df_subset[output_list]
        outputs = outputs.loc[:, ~outputs.columns.duplicated()]
        col_rename_dict = {c: century_to_rp(c) for c in outputs.columns.values}
        outputs.rename(index=int, columns=col_rename_dict, inplace=True)
        outputs[sbstr] = 1
        if sbstr == 'site':
            outputs.to_csv(site_initial_table, index=False)
        if sbstr == 'PFT':
            outputs.to_csv(pft_initial_table, index=False)


def century_params_to_new_model_params(
        pft_param_path, animal_param_path, site_param_path):
    """Generate parameter inputs for the new forage model.

    Site and pft parameters for the new forage model come from various
    files used by Century.  Gather these parameters together from all
    Century parameter files and format them as csvs as expected by new
    forage model.

    Parameters:
        pft_param_path (string): path where the pft parameter table for RPM
            should be saved
        animal_param_path (string): path where the animal parameter table for
            RPM should be saved
        site_param_path (string): path where the site parameter table for RPM
            should be saved

    Returns:
        None

    """
    CENTURY_DIR = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014"
    TEMPLATE_HIST = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015/0_hist.sch"
    TEMPLATE_SCH = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015/0.sch"
    TEMPLATE_100 = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Kenya/input/regional_properties/Worldclim_precip/empty_2014_2015/0.100"
    new_model_args = {
        'template_level': 'GLP',
        'fix_file': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_inputs/drytrpfi.100",
        'grass_type': 'C4',
        'herbivore_csv': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/herd_avg_uncalibrated.csv"
    }
    # parameter table containing only necessary parameters
    parameter_table = pandas.read_csv(
        "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/GK_doc/Century_parameter_table.csv")
    parameters_to_keep = parameter_table['Century parameter name'].tolist()
    crop_params = os.path.join(CENTURY_DIR, 'crop.100')

    # get crop from TEMPLATE_HIST and TEMPLATE_SCH
    first_month = set()
    senescence_month = set()
    last_month = set()
    crop_list = set()
    with open(TEMPLATE_HIST, 'r') as hist_sch:
        for line in hist_sch:
            if ' CROP' in line:
                crop_line = next(hist_sch)
                crop_list.add(crop_line[:10].strip())
            if ' FRST' in line:
                first_month.add(line[7:10].strip())
            if ' SENM' in line:
                senescence_month.add(line[7:10].strip())
            if ' LAST' in line:
                last_month.add(line[7:10].strip())
    with open(TEMPLATE_SCH, 'r') as hist_sch:
        for line in hist_sch:
            if ' CROP' in line:
                crop_line = next(hist_sch)
                crop_list.add(crop_line[:10].strip())
            if ' FRST' in line:
                first_month.add(line[7:10].strip())
            if ' SENM' in line:
                senescence_month.add(line[7:10].strip())
            if ' LAST' in line:
                last_month.add(line[7:10].strip())
    # ensure that crop (e.g. GCD_G) is same between hist and extend schedule
    assert len(crop_list) == 1, "We can only handle one PFT for old model"
    # ensure that the file contains only one schedule to begin and end
    # growth
    assert len(first_month) == 1, "More than one starting month found"
    assert len(last_month) == 1, "More than one ending month found"
    PFT_label = list(crop_list)[0]

    # collect parameters from all Century sources
    master_param_dict = {}
    # get crop parameters from crop.100 file in CENTURY_DIR
    with open(crop_params, 'r') as cparam:
        for line in cparam:
            if line.startswith('{} '.format(PFT_label)):
                while 'MXDDHRV' not in line:
                    label = re.sub(r"\'", "", line[13:].strip()).lower()
                    if label in parameters_to_keep:
                        value = float(line[:13].strip())
                        master_param_dict[label] = value
                    line = next(cparam)
                label = re.sub(r"\'", "", line[13:].strip()).lower()
                if label in parameters_to_keep:
                    value = float(line[:13].strip())
                    master_param_dict[label] = value
    # get grazing effect parameters from graz.100 file
    graz_file = os.path.join(CENTURY_DIR, 'graz.100')
    with open(graz_file, 'r') as grazparams:
        for line in grazparams:
            if line.startswith(new_model_args['template_level']):
                line = next(grazparams)
                while 'FECLIG' not in line:
                    label = re.sub(r"\'", "", line[13:].strip()).lower()
                    if label in parameters_to_keep:
                        value = float(line[:13].strip())
                        master_param_dict[label] = value
                    line = next(grazparams)
                label = re.sub(r"\'", "", line[13:].strip()).lower()
                if label in parameters_to_keep:
                    value = float(line[:13].strip())
                    master_param_dict[label] = value
    # get site parameters from TEMPLATE_100
    with open(TEMPLATE_100, 'r') as siteparam:
        for line in siteparam:
            label = re.sub(r"\'", "", line[13:].strip()).lower()
            if label in parameters_to_keep:
                value = float(line[:13].strip())
                master_param_dict[label] = value
    # get fixed parameters from new_model_args['fix_file']
    with open(new_model_args['fix_file'], 'r') as siteparam:
        for line in siteparam:
            label = re.sub(r"\'", "", line[13:].strip()).lower()
            if label in parameters_to_keep:
                value = float(line[:13].strip())
                master_param_dict[label] = value

    def century_to_rp(century_label):
        """Convert Century name to rangeland production name."""
        rp = re.sub(r"\(", "_", century_label)
        rp = re.sub(r",", "_", rp)
        rp = re.sub(r"\)", "", rp)
        return rp

    # apportion parameters to PFT and site tables
    PFT_param_dict = {'PFT': 1}
    pft_params = parameter_table[
        parameter_table['Property of'] == 'PFT']['Century parameter name']
    for label in pft_params:
        PFT_param_dict[label] = master_param_dict[label]
    site_param_dict = {'site': 1}
    site_params = parameter_table[
        parameter_table['Property of'] == 'site']['Century parameter name']
    for label in site_params:
        site_param_dict[label] = master_param_dict[label]
    animal_param_dict = {'animal_id': 1}
    animal_params = parameter_table[
        parameter_table['Property of'] == 'animal']['Century parameter name']
    for label in animal_params:
        animal_param_dict[label] = master_param_dict[label]

    # add to grass csv to make PFT trait table
    PFT_param_dict['growth_months'] = (
        [','.join([str(m) for m in range(
            int(list(first_month)[0]), int(list(last_month)[0]) + 1)])])
    if senescence_month:
        PFT_param_dict['senescence_month'] = (
            ','.join([str(m) for m in list(senescence_month)]))
    if new_model_args['grass_type'] == 'C3':
        PFT_param_dict['species_factor'] = 0
    else:
        PFT_param_dict['species_factor'] = 0.16
    pft_df = pandas.DataFrame(PFT_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in pft_df.columns.values}
    pft_df.rename(index=int, columns=col_rename_dict, inplace=True)
    pft_df.to_csv(pft_param_path, index=False)
    # TODO: add new PFT parameters:
    #   digestibility_slope
    #   digestibility_intercept

    # add to herbivore csv to make animal parameter table
    animal_beta_df = pandas.read_csv(new_model_args['herbivore_csv'])
    animal_df = pandas.DataFrame(animal_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in animal_df.columns.values}
    animal_df.rename(index=int, columns=col_rename_dict, inplace=True)
    merged_animal_df = pandas.concat(
        [animal_beta_df, animal_df], axis=1, sort=False)
    merged_animal_df.to_csv(animal_param_path, index=False)
    # TODO: add parameter 'grzeff'

    # make site parameter table
    site_df = pandas.DataFrame(site_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in site_df.columns.values}
    site_df.rename(index=int, columns=col_rename_dict, inplace=True)
    site_df.to_csv(site_param_path, index=False)


def wcs_monitoring_points_workflow():
    """Run Century to get initialization rasters for WCS monitoring area."""
    # site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/soil/monitoring_points_soil_isric_250m.csv"
    # input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/chirps_prec_back_calc"
    # outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/monitoring_sites/chirps_prec_back_calc_2.13.20"
    # # launch_sites(site_csv, input_dir, outer_outdir)
    # year = 2016
    # month = 8
    # site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/site_idx_CBM_SCP_voronoi_polygon.tif"
    # initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/initial_conditions"
    # century_outputs_to_rpm_initial_rasters(
    #     site_csv, outer_outdir, year, month, site_index_path,
    #     initial_conditions_dir)
    # start_time = 2015.08
    # end_time = 2018.00
    # save_as = os.path.join(outer_outdir, 'biomass_summary.csv')
    # summarize_century_biomass(
    #     site_csv, outer_outdir, start_time, end_time, save_as)
    # initialization for scenarios in monitoring area
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/worldclim_historical_sch/intermediate_data/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/worldclim_historical_sch/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/monitoring_sites/worldclim_historical_sch"
    launch_sites(site_csv, 'site_id', input_dir, outer_outdir)
    year = 2016
    month = 12
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/site_idx_CBM_SCP_voronoi_polygon.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/worldclim_historical_sch/RPM_inputs/initial_conditions"
    century_outputs_to_rpm_initial_rasters(
        site_csv, 'site_id', outer_outdir, year, month, site_index_path,
        initial_conditions_dir, 'site_id_num')


def julian_ahlborn_sites_workflow():
    """Run Century to get initialization rasters for Julian Ahlborn's sites."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_inputs_Aug2020"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_outputs_Aug2020"
    # launch_sites(site_csv, 'site', input_dir, outer_outdir)
    start_time = 2014.08
    end_time = 2015.00
    save_as = os.path.join(outer_outdir, 'biomass_summary.csv')
    # summarize_century_biomass(
        # site_csv, 'site', outer_outdir, start_time, end_time, save_as)
    year = 2011
    month = 8
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_index.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/initial_conditions_Aug20"
    # century_outputs_to_rpm_initial_rasters(
    #     site_csv, 'site', outer_outdir, year, month, site_index_path,
    #     initial_conditions_dir)
    # Nov 2020, collect biomass at site centroids with no grazing
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_inputs_Aug2020/no_grazing_last_year"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_outputs_Aug2020/no_grazing_last_year"
    launch_sites(site_csv, 'site', input_dir, outer_outdir)
    start_time = 2015.08
    end_time = 2016.00
    save_as = os.path.join(outer_outdir, 'biomass_summary.csv')
    summarize_century_biomass(
        site_csv, 'site', outer_outdir, start_time, end_time, save_as)


def n_content_experiment():
    """Run Century at one of Julian Ahlborn's sites, experimenting with
    parameters controlling N content of new production."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil_site15.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites_n_content_exp/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites_n_content_exp/Century_outputs"
    launch_sites(site_csv, 'site', input_dir, outer_outdir)
    # calculate crude protein content
    min_time = 2016.00
    max_time = 2017.00
    century_output_file = os.path.join(outer_outdir, '15', '15.lis')
    cent_df = pandas.read_fwf(century_output_file, skiprows=[1])
    cent_df = cent_df[(cent_df.time >= min_time) & (cent_df.time <= max_time)]
    cp_aglivc = (cent_df['aglive(1)'] * 6.25) / (cent_df['aglivc'] * 2.5)
    print(cp_aglivc)
    print('\nmean cp: {}'.format(cp_aglivc.mean()))
    print(cent_df['aglivc'])
    print('\nmean aglivc: {}'.format(cent_df['aglivc'].mean()))
    shutil.rmtree("C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites_n_content_exp/Century_outputs/15")


def eastern_steppe_initial_tables():
    century_output_file = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_outputs_Aug2020/12/12.lis"
    year = 2014
    month = 12
    site_initial_table = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/site_initial_table.csv"
    pft_initial_table = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/pft_initial_table.csv"
    century_outputs_to_initial_tables(
        century_output_file, year, month, site_initial_table,
        pft_initial_table)


def eastern_steppe_npp():
    """Run Century and collect "crproda", NPP."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/Century_inputs/zerosd"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/Century_outputs/zerosd"
    launch_sites(site_csv, 'id', input_dir, outer_outdir)
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/site_index.tif"
    target_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/Century_outputs/zerosd/average_NPP_zerosd.tif"
    century_npp_to_raster(
        site_csv, 'id', outer_outdir, site_index_path, target_path)


def ahlborn_aoi_npp():
    """Run Century and collect "cproda", NPP."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_halfdegree_grid/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_halfdegree_grid/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_halfdegree_grid/Century_outputs"
    launch_sites(site_csv, 'id', input_dir, outer_outdir)
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_halfdegree_grid/site_index.tif"
    target_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_halfdegree_grid/Century_outputs/average_NPP_zerosd.tif"
    century_npp_to_raster(
        site_csv, 'id', outer_outdir, site_index_path, target_path)


def ahlborn_scenario_initial_rasters():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil.csv"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_sites/Century_outputs_Aug2020"
    year = 2011
    month = 12
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_index.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/initial_conditions_scenarios"
    century_outputs_to_rpm_initial_rasters(
        site_csv, 'site', outer_outdir, year, month, site_index_path,
        initial_conditions_dir)

def laikipia_scenario_initial_rasters():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_outputs"
    # launch_sites(
    #     site_csv, 'id', input_dir, outer_outdir, fix_file='drytrpfi.100')
    year = 2013
    month = 12
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/voronoi_polygons_regular_grid.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/initial_conditions"
    # century_outputs_to_rpm_initial_rasters(
    #     site_csv, 'id', outer_outdir, year, month, site_index_path,
    #     initial_conditions_dir)
    pft_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/pft_parameter_table.csv"
    animal_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/animal_parameter_table.csv"
    site_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/site_parameter_table.csv"
    century_params_to_new_model_params(
        pft_param_path, animal_param_path, site_param_path)


def laikipia_npp():
    """Run Century and collect "cproda", NPP."""
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_outputs_NPP"
    # launch_sites(
        # site_csv, 'id', input_dir, outer_outdir, fix_file='drytrpfi.100')
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/voronoi_polygons_regular_grid.tif"
    target_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/Century_outputs_NPP/average_NPP_zerosd.tif"
    century_npp_to_raster(
        site_csv, 'id', outer_outdir, site_index_path, target_path)


def OPC_initial_rasters():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/Century_outputs"
    # launch_sites(
    #     site_csv, 'id', input_dir, outer_outdir, fix_file='drytrpfi.100')
    year = 2013
    month = 12
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/voronoi_polygons_regular_grid.tif"
    initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/RPM_inputs/initial_conditions"
    century_outputs_to_rpm_initial_rasters(
        site_csv, 'id', outer_outdir, year, month, site_index_path,
        initial_conditions_dir)
    pft_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/RPM_inputs/pft_parameter_table.csv"
    animal_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/RPM_inputs/animal_parameter_table.csv"
    site_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/RPM_inputs/site_parameter_table.csv"
    century_params_to_new_model_params(
        pft_param_path, animal_param_path, site_param_path)


def eastern_steppe_regular_grid_initial_rasters():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/Century_outputs"
    # launch_sites(site_csv, 'id', input_dir, outer_outdir)
    # year = 2011
    # month = 12
    # site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/site_index.tif"
    # initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/initial_conditions"
    # century_outputs_to_rpm_initial_rasters(
    #     site_csv, 'id', outer_outdir, year, month, site_index_path,
    #     initial_conditions_dir)
    year = 2011
    month = 18
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/site_index.tif"
    initial_conditions_dir = "C:/Users/ginge/Desktop/Century_Worldclim"
    century_outputs_to_rpm_initial_rasters(
        site_csv, 'id', outer_outdir, year, month, site_index_path,
        initial_conditions_dir)


def eastern_steppe_regular_grid_future_climate():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/Century_inputs_CanESM5"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/Century_outputs_CanESM5"
    # launch_sites(site_csv, 'id', input_dir, outer_outdir)
    # year = 2011
    # month = 12
    # site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/site_index.tif"
    # initial_conditions_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/initial_conditions_CanESM5"
    # century_outputs_to_rpm_initial_rasters(
    #     site_csv, 'id', outer_outdir, year, month, site_index_path,
    #     initial_conditions_dir)
    year = 2011
    month = 8
    site_index_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/site_index.tif"
    initial_conditions_dir = "C:/Users/ginge/Desktop/Century_CanESM5"
    century_outputs_to_rpm_initial_rasters(
        site_csv, 'id', outer_outdir, year, month, site_index_path,
        initial_conditions_dir)


def Century_time_series():
    site_csv = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/time_series_points/soil.csv"
    input_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/time_series_points/Century_inputs"
    outer_outdir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/time_series_points/Century_outputs_CanESM5"
    launch_sites(site_csv, 'id', input_dir, outer_outdir)\


def main():
    """Program entry point."""
    # julian_ahlborn_sites_workflow()
    # wcs_monitoring_points_workflow()
    # eastern_steppe_initial_tables()
    # n_content_experiment()
    # ahlborn_scenario_initial_rasters()
    # laikipia_scenario_initial_rasters()
    # OPC_initial_rasters()
    # eastern_steppe_regular_grid_initial_rasters()
    # eastern_steppe_regular_grid_future_climate()
    # Century_time_series()
    # laikipia_npp()
    # eastern_steppe_npp()
    ahlborn_aoi_npp()


if __name__ == "__main__":
    main()
