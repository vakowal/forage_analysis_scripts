"""Generate regression testing results for new forage model.

This module contains functions to generate regression testing rasters from
the old forage model, which can be used to test the new forage model.
From a given set of data that are sufficient to launch the new forage model
(i.e. the model containing a Century implementation in Python), generate
inputs for the old forage model, including inputs for a series of points to
run the original Fortran version of Century in a grid, emulating a raster.
Collect results from the old model and convert them to raster, to be used
as regression tests for the new model.
"""
import os
import sys
import collections
# import taskgraph
import tempfile
from tempfile import mkstemp
import shutil
import re
import math

import pandas
import numpy
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
import pygeoprocessing
# import arcpy

sys.path.append("C:/Users/ginge/Documents/Python/rangeland_production")
# import forage as old_model

# arcpy.CheckOutExtension("Spatial")

SAMPLE_DATA = r"C:\Users\ginge\Dropbox\sample_inputs"
TEMPLATE_100 = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\no_grazing.100"
TEMPLATE_HIST = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\historical_schedule_hist.sch"
TEMPLATE_SCH = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\historical_schedule.sch"
CENTURY_DIR = 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
DROPBOX_DIR = "C:/Users/ginge/Dropbox/NatCap_backup"
LOCAL_DIR = "C:/Users/ginge/Documents/NatCap/GIS_local"
GRID_POINT_SHP = os.path.join(LOCAL_DIR, "raster_template_point.shp")


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
    outvar_csv = os.path.join(
        DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
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


def convert_to_year_month(CENTURY_date):
    """Convert CENTURY's representation of dates (from output file) to year
    and month.  Returns a list containing integer year and integer month."""

    CENTURY_date = float(CENTURY_date)
    if CENTURY_date - math.floor(CENTURY_date) == 0:
        year = int(CENTURY_date - 1)
        month = 12
    else:
        year = int(math.floor(CENTURY_date))
        month = int(round(12. * (CENTURY_date - year)))
    return [year, month]


def convert_to_century_date(year, month):
    """Convert year and month to Century's decimal representation of dates."""
    return float('%.2f' % (year + month / 12.))


def check_raster_dimensions(raster1_path, raster2_path):
    """Ensure that raster1 and raster2 have identical dimensions."""
    base_raster_path_band_const_list = [
        (raster1_path, 1), (raster2_path, 1)]
    raster_info_list = [
        pygeoprocessing.get_raster_info(path_band[0])
        for path_band in base_raster_path_band_const_list
        if pygeoprocessing._is_raster_path_band_formatted(path_band)]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))


def generate_site_shp_from_raster(base_raster_path, target_point_vector_path):
    """Make a point vector from a raster.

    Old Century is launched for a series of individual points, but the new
    model is launched from rasters. Generate the point vector shapefile that is
    used to index raster pixels to individual points.  This shapefile is used
    to extract point-based inputs from raster-based inputs, and again to
    generate raster-based results, which can be used to initialize or
    regression test the new model, from a series of point-based results from
    the old model. Point geometry is centered on the center of the pixel
    square.

    Parameters:
        base_raster_path (string): path to a single band template raster.
        target_point_vector_path (string): path to a point vector that will
            contain points which are centered on each pixel box for as many
            pixels in the raster. The feature attributes for each point will
            include:
                'x_coord': original x pixel center point in projected units of
                    `base_raster_path`.
                'y_coord': original y pixel center point in projected units of
                    `base_raster_path`.
                'raster_x': x coordinate of the raster pixel
                'raster_y': y coordinate of the raster pixel
                'site_id': unique integer for each point.

    Returns:
        None.
    """
    if os.path.exists(target_point_vector_path):
        os.remove(target_point_vector_path)
    esri_shapefile_driver = ogr.GetDriverByName('ESRI Shapefile')
    target_vector = esri_shapefile_driver.CreateDataSource(
        target_point_vector_path)
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    target_layer = target_vector.CreateLayer(
        target_point_vector_path,
        srs=osr.SpatialReference(base_raster_info['projection']),
        geom_type=ogr.wkbPoint)
    target_layer.CreateField(ogr.FieldDefn("x_coord", ogr.OFTReal))
    target_layer.CreateField(ogr.FieldDefn("y_coord", ogr.OFTReal))
    target_layer.CreateField(ogr.FieldDefn("raster_x", ogr.OFTInteger))
    target_layer.CreateField(ogr.FieldDefn("raster_y", ogr.OFTInteger))
    target_layer.CreateField(ogr.FieldDefn("site_id", ogr.OFTInteger))

    # Create the feature and set values
    feature_defn = target_layer.GetLayerDefn()

    geotransform = base_raster_info['geotransform']
    for offset_map, raster_block in pygeoprocessing.iterblocks(
            (base_raster_path, 1)):
        n_y_block = raster_block.shape[0]
        n_x_block = raster_block.shape[1]

        # offset by .5 so we're in the center of the pixel
        xoff = offset_map['xoff'] + 0.5
        yoff = offset_map['yoff'] + 0.5

        # calculate the projected x and y coordinate bounds for the block
        x_range = numpy.linspace(
            geotransform[0] + geotransform[1] * xoff,
            geotransform[0] + geotransform[1] * (xoff + n_x_block - 1),
            n_x_block)
        y_range = numpy.linspace(
            geotransform[3] + geotransform[5] * yoff,
            geotransform[3] + geotransform[5] * (yoff + n_y_block - 1),
            n_y_block)

        # these are the x and y coordinate bounds for the pixel indexes
        x_index_range = numpy.linspace(
            xoff, xoff + (n_x_block - 1), n_x_block)
        y_index_range = numpy.linspace(
            yoff, yoff + (n_y_block - 1), n_y_block)

        # we'll use this to avoid generating any nodata points
        valid_mask = raster_block != base_raster_info['nodata']

        # these indexes correspond to projected coordinates
        x_vector, y_vector = numpy.meshgrid(x_range, y_range)
        # these correspond to raster indexes
        xi_vector, yi_vector = numpy.meshgrid(x_index_range, y_index_range)

        site_id = 0
        for value, x_coord, y_coord, raster_x, raster_y in zip(
                raster_block[valid_mask],
                x_vector[valid_mask], y_vector[valid_mask],
                xi_vector[valid_mask], yi_vector[valid_mask]):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(x_coord, y_coord)
            feature = ogr.Feature(feature_defn)
            feature.SetGeometry(point)
            feature.SetField("x_coord", x_coord)
            feature.SetField("y_coord", y_coord)
            feature.SetField("raster_x", raster_x)
            feature.SetField("raster_y", raster_y)
            feature.SetField("site_id", site_id)
            target_layer.CreateFeature(feature)
            feature = None
            site_id += 1


def generate_base_args():
    """Inputs to run both new and old model."""
    args = {
            'starting_month': 1,
            'starting_year': 2016,
            'n_months': 12,
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'aoi_small.shp'),
            'bulk_density_path': os.path.join(
                SAMPLE_DATA, 'soil', 'bulkd.tif'),
            'ph_path': os.path.join(
                SAMPLE_DATA, 'soil', 'phihox_sl3.tif'),
            'clay_proportion_path': os.path.join(
                SAMPLE_DATA, 'soil', 'clay.tif'),
            'silt_proportion_path': os.path.join(
                SAMPLE_DATA, 'soil', 'silt.tif'),
            'sand_proportion_path': os.path.join(
                SAMPLE_DATA, 'soil', 'sand.tif'),
            'monthly_precip_path_pattern': os.path.join(
                SAMPLE_DATA, 'CHIRPS_div_by_10',
                'chirps-v2.0.<year>.<month>.tif'),
            'min_temp_path_pattern': os.path.join(
                SAMPLE_DATA, 'temp', 'wc2.0_30s_tmin_<month>.tif'),
            'max_temp_path_pattern': os.path.join(
                SAMPLE_DATA, 'temp', 'wc2.0_30s_tmax_<month>.tif'),
            'site_param_path': os.path.join(
                SAMPLE_DATA, 'site_parameters.csv'),
            'site_param_spatial_index_path': os.path.join(
                SAMPLE_DATA, 'site_index.tif'),
            'veg_trait_path': os.path.join(SAMPLE_DATA, 'pft_trait.csv'),
            'veg_spatial_composition_path_pattern': os.path.join(
                SAMPLE_DATA, 'pft<PFT>.tif'),
            'animal_trait_path': os.path.join(
                SAMPLE_DATA, 'animal_trait_table.csv'),
            'animal_mgmt_layer_path': os.path.join(
                SAMPLE_DATA, 'sheep_units_density_2016_monitoring_area.shp'),
            'workspace_dir': SAMPLE_DATA,
            'site_table': os.path.join(
                DROPBOX_DIR,
                "Mongolia/model_inputs/pycentury_dev/soil_table.csv"),
            'grass_csv': os.path.join(
                DROPBOX_DIR, "Mongolia/model_inputs/grass.csv"),
            'herbivore_csv': os.path.join(
                DROPBOX_DIR, "Mongolia/model_inputs/cashmere_goats.csv"),
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
        }
    return args


def generate_aligned_inputs():
    """Align and resize raster inputs using same methods as in new model.

    In the new model, raster-based inputs must be aligned and resampled to
    ensure that they align exactly.  Before generating point-based inputs for
    the old model, align and resample raw inputs using the same methods that
    are used internally in the new model.

    This code copied from (new) forage.py, lines 345-594.

    Returns:
        aligned_args, a full dict of inputs to run the new model, including
            paths to raster inputs where all rasters have been aligned
    """
    args_new_model = generate_base_args()
    aligned_args = args_new_model.copy()

    # set up a dictionary that uses the same keys as
    # 'base_align_raster_path_id_map' to point to the clipped/resampled
    # rasters
    aligned_raster_dir = os.path.join(
        args_new_model['workspace_dir'], 'aligned_inputs')
    if not os.path.exists(aligned_raster_dir):
        os.makedirs(aligned_raster_dir)

    for arg_key in [
            'bulk_density_path', 'ph_path', 'clay_proportion_path',
            'silt_proportion_path', 'sand_proportion_path',
            'monthly_precip_path_pattern', 'min_temp_path_pattern',
            'max_temp_path_pattern', 'site_param_spatial_index_path',
            'veg_spatial_composition_path_pattern']:
        aligned_args[arg_key] = os.path.join(
            aligned_raster_dir, os.path.basename(args_new_model[arg_key]))

    starting_month = int(args_new_model['starting_month'])
    starting_year = int(args_new_model['starting_year'])
    n_months = int(args_new_model['n_months'])

    # collect precipitation inputs
    temperature_month_set = set()
    base_align_raster_path_id_map = {}
    for month_index in xrange(n_months):
        month_i = (starting_month + month_index - 1) % 12 + 1
        temperature_month_set.add(month_i)
        year = starting_year + (starting_month + month_index - 1) // 12
        precip_path = args_new_model[
            'monthly_precip_path_pattern'].replace(
                '<year>', str(year)).replace('<month>', '%.2d' % month_i)
        base_align_raster_path_id_map['precip_%d' % month_index] = precip_path

    # collect temperature inputs
    for substring in ['min', 'max']:
        for month_i in temperature_month_set:
            monthly_temp_path = args_new_model[
                '%s_temp_path_pattern' % substring].replace(
                    '<month>', '%.2d' % month_i)
            base_align_raster_path_id_map[
                '%s_temp_%d' % (substring, month_i)] = monthly_temp_path

    # soil inputs
    for soil_type in ['clay', 'silt', 'sand']:
        base_align_raster_path_id_map[soil_type] = (
            args_new_model['%s_proportion_path' % soil_type])
    base_align_raster_path_id_map['bulk_d_path'] = args_new_model[
        'bulk_density_path']
    base_align_raster_path_id_map['ph_path'] = args_new_model['ph_path']

    # site and pft inputs
    base_align_raster_path_id_map['site_index'] = (
        args_new_model['site_param_spatial_index_path'])
    pft_dir = os.path.dirname(args_new_model[
        'veg_spatial_composition_path_pattern'])
    pft_basename = os.path.basename(
        args_new_model['veg_spatial_composition_path_pattern'])
    files = [
        f for f in os.listdir(pft_dir) if os.path.isfile(
            os.path.join(pft_dir, f))]
    pft_regex = re.compile(pft_basename.replace('<PFT>', r'(\d+)'))
    pft_matches = [
        m for m in [pft_regex.search(f) for f in files] if m is not None]
    pft_id_set = set([int(m.group(1)) for m in pft_matches])
    for pft_i in pft_id_set:
        pft_path = args_new_model[
            'veg_spatial_composition_path_pattern'].replace(
            '<PFT>', '%d' % pft_i)
        base_align_raster_path_id_map['pft_%d' % pft_i] = pft_path

    # find the smallest target_pixel_size
    # TODO uncomment this to return to default of smallest pixel size
    # target_pixel_size = min(*[
    #     pygeoprocessing.get_raster_info(path)['pixel_size']
    #     for path in base_align_raster_path_id_map.values()],
    #     key=lambda x: (abs(x[0]), abs(x[1])))
    # TODO remove this; hacking in running model at CHIRPS resolution
    target_pixel_size = pygeoprocessing.get_raster_info(
        base_align_raster_path_id_map['precip_0'])['pixel_size']

    # align all the base inputs to be the minimum known pixel size and to
    # only extend over their combined intersections
    source_input_path_list = [
        base_align_raster_path_id_map[k] for k in sorted(
            base_align_raster_path_id_map.iterkeys())]
    aligned_input_path_list = [
        os.path.join(aligned_raster_dir, os.path.basename(path)) for path in
        source_input_path_list]

    if all([os.path.exists(p) for p in aligned_input_path_list]):
        return aligned_args

    pygeoprocessing.align_and_resize_raster_stack(
        source_input_path_list, aligned_input_path_list,
        ['near'] * len(aligned_input_path_list),
        target_pixel_size, 'intersection',
        base_vector_path_list=[args_new_model['aoi_path']])
    return aligned_args


def generate_inputs_for_old_model(processing_dir, input_dir):
    """Generate inputs for the old forage model.

    The old model uses the Century executable written in Fortran. Its inputs
    pertain to a single point. To emulate a gridded raster, we can run the
    model for a series of points in a regular grid.  Generate the inputs to
    run the model in a gridded series of points.

    Returns:
        None
    """
    def write_soil_table(site_shp, save_as):
        """Make soil table to use as input for site files."""
        # make a temporary copy of the point shapefile to append
        # worldclim values
        tempdir = tempfile.mkdtemp()
        source_shp = site_shp
        site_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, site_shp)

        raster_files = [aligned_args['bulk_density_path'],
                        aligned_args['ph_path'],
                        aligned_args['clay_proportion_path'],
                        aligned_args['silt_proportion_path'],
                        aligned_args['sand_proportion_path']]
        field_list = [os.path.basename(r)[:-4] for r in raster_files]
        ex_list = zip(raster_files, field_list)
        arcpy.sa.ExtractMultiValuesToPoints(site_shp, ex_list)

        # read from shapefile to new table
        field_list.insert(0, 'site_id')
        temp_dict = {field: [] for field in field_list}
        with arcpy.da.SearchCursor(site_shp, field_list) as cursor:
            for row in cursor:
                for f_idx in range(len(field_list)):
                    temp_dict[field_list[f_idx]].append(row[f_idx])
        soil_df = pandas.DataFrame.from_dict(temp_dict).set_index('site_id')

        bulkd_key = os.path.basename(
            aligned_args['bulk_density_path'])[:-4][:10]
        ph_key = os.path.basename(aligned_args['ph_path'])[:-4][:10]
        clay_key = os.path.basename(
            aligned_args['clay_proportion_path'])[:-4][:10]
        silt_key = os.path.basename(
            aligned_args['silt_proportion_path'])[:-4][:10]
        sand_key = os.path.basename(
            aligned_args['sand_proportion_path'])[:-4][:10]
        # check units
        soil_df = soil_df[soil_df[bulkd_key] > 0]
        # bulk density should be between 0.8 and 2
        if soil_df[bulkd_key].mean() > 9:
            soil_df[bulkd_key] = soil_df[bulkd_key] / 10.
        elif soil_df[bulkd_key].mean() < 0.9:
            soil_df[bulkd_key] = soil_df[bulkd_key] * 10.
        assert min(soil_df[bulkd_key]) > 0.8, "Bulk density out of range"
        assert max(soil_df[bulkd_key]) < 2, "Bulk density out of range"
        # pH should be between 5.9 and 9
        if soil_df[ph_key].mean() > 9:
            soil_df[ph_key] = soil_df[ph_key] / 10.
        elif soil_df[ph_key].mean() < 0.9:
            soil_df[ph_key] = soil_df[ph_key] * 10.
        assert min(soil_df[ph_key]) > 5.9, "pH out of range"
        assert max(soil_df[ph_key]) < 9., "pH out of range"
        # clay, sand and silt should be between 0 and 1
        while min(soil_df[clay_key]) > 1:
            soil_df[clay_key] = soil_df[clay_key] / 10.
        assert max(soil_df[clay_key]) <= 1., "clay out of range"
        while min(soil_df[silt_key]) > 1:
            soil_df[silt_key] = soil_df[silt_key] / 10.
        assert max(soil_df[silt_key]) <= 1., "silt out of range"
        while min(soil_df[sand_key]) > 1:
            soil_df[sand_key] = soil_df[sand_key] / 10.
        assert max(soil_df[sand_key]) <= 1., "sand out of range"

        # add lat and long
        soil_df["latitude"] = "NA"
        soil_df["longitude"] = "NA"
        # WGS 1984
        arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)
        arcpy.AddXY_management(site_shp)
        with arcpy.da.SearchCursor(
                site_shp, ['site_id', 'POINT_X', 'POINT_Y']) as cursor:
            for row in cursor:
                try:
                    soil_df = soil_df.set_value([row[0]], 'longitude', row[1])
                    soil_df = soil_df.set_value([row[0]], 'latitude', row[2])
                except KeyError:
                    continue
        soil_df.to_csv(save_as)


    def write_temperature_table(site_shp, save_as):
        """Make a table of max and min monthly temperature for points."""
        tempdir = tempfile.mkdtemp()
        # make a temporary copy of the point shapefile to append
        # worldclim values
        source_shp = site_shp
        point_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, point_shp)

        temperature_month_set = set()
        starting_month = int(aligned_args['starting_month'])
        for month_index in xrange(int(aligned_args['n_months'])):
            month_i = (starting_month + month_index - 1) % 12 + 1
            temperature_month_set.add(month_i)

        raster_files = []
        field_list = []
        for substring in ['min', 'max']:
            for month_i in temperature_month_set:
                monthly_temp_path = aligned_args[
                    '%s_temp_path_pattern' % substring].replace(
                    '<month>', '%.2d' % month_i)
                raster_files.append(monthly_temp_path)
                field_list.append('{}_{}'.format(substring, month_i))
        ex_list = zip(raster_files, field_list)
        arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

        # read from shapefile to newly formatted table
        field_list.insert(0, 'site_id')
        temp_dict = {'site': [], 'month': [], 'tmin': [], 'tmax': []}
        with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
            for row in cursor:
                site = row[0]
                temp_dict['site'].extend([site] * len(temperature_month_set))
                for f_idx in range(1, len(field_list)):
                    field = field_list[f_idx]
                    if field.startswith('min'):
                        temp_dict['tmin'].append(row[f_idx])
                    elif field.startswith('max'):
                        temp_dict['tmax'].append(row[f_idx])
                    else:
                        raise ValueError("value not recognized")
                temp_dict['month'].extend(temperature_month_set)
        for key in temp_dict.keys():
            if len(temp_dict[key]) == 0:
                del temp_dict[key]
        temp_df = pandas.DataFrame.from_dict(temp_dict)
        # check units
        while max(temp_df['tmax']) > 100:
            temp_df['tmax'] = temp_df['tmax'] / 10.
            temp_df['tmin'] = temp_df['tmin'] / 10.
        temp_df.to_csv(save_as, index=False)


    def write_worldclim_precip_table(site_shp, save_as):
        """Write precipitation table from Worldclim average precipitation.

        Worldclim average precipitation should be used for spin-up simulations.
        """
        worldclim_pattern = os.path.join(
            LOCAL_DIR, "Mongolia/Worldclim/precip/wc2.0_10m_prec_<month>.tif")

        tempdir = tempfile.mkdtemp()
        # make a temporary copy of the point shapefile to append
        # worldclim values
        source_shp = site_shp
        point_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, point_shp)

        precip_dir = os.path.dirname(worldclim_pattern)
        precip_basename = os.path.basename(worldclim_pattern)
        files = [f for f in os.listdir(precip_dir) if os.path.isfile(
                 os.path.join(precip_dir, f))]
        precip_regex = re.compile(precip_basename.replace('<month>', '(\d+)'))
        precip_matches = [m for m in [precip_regex.search(f) for f in files]
                          if m is not None]
        month_list = set([int(m.group(1)) for m in precip_matches])
        raster_files = []
        for month_i in month_list:
            precip_path = worldclim_pattern.replace('<month>',
                                                    '%02d' % month_i)
            raster_files.append(precip_path)
        ex_list = zip(raster_files, month_list)
        arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

        # read from shapefile to newly formatted table
        field_list = [str(m) for m in month_list]
        field_list.insert(0, 'site_id')
        prec_dict = {'site': [], 'month': [], 'prec': []}
        with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
            for row in cursor:
                site = row[0]
                prec_dict['site'].extend([site] * 12)
                for f_idx in range(1, len(field_list)):
                    field = field_list[f_idx]
                    month = int(field)
                    prec_dict['prec'].append(row[f_idx])
                prec_dict['month'].extend(month_list)
        prec_df = pandas.DataFrame.from_dict(prec_dict)
        # divide raw Worldclim precip by 10
        prec_df['prec'] = prec_df['prec'] / 10.
        prec_df.to_csv(save_as, index=False)


    def write_precip_table_from_rasters(aligned_args, site_shp, save_as):
        """Make a table of precipitation from a series of rasters.

        This adapted from Rich's script point_precip_fetch.py.
        """
        # task_graph = taskgraph.TaskGraph('taskgraph_cache', 0)
        # task_graph.close()
        # task_graph.join()

        starting_month = int(aligned_args['starting_month'])
        starting_year = int(aligned_args['starting_year'])
        n_months = int(aligned_args['n_months'])

        # get precip rasters from args dictionary, index each by their basename
        raster_path_id_map = {}
        for month_index in xrange(n_months):
            month_i = (starting_month + month_index - 1) % 12 + 1
            year = starting_year + (starting_month + month_index - 1) // 12
            precip_path = aligned_args[
                'monthly_precip_path_pattern'].replace(
                    '<year>', str(year)).replace('<month>', '%.2d' % month_i)
            basename = os.path.basename(precip_path)
            raster_path_id_map[basename] = precip_path

        # pre-fetch point geometry and IDs
        point_vector = ogr.Open(site_shp)
        point_layer = point_vector.GetLayer()
        point_defn = point_layer.GetLayerDefn()

        # build up a list of the original field names so we can copy it to
        # report
        point_field_name_list = []
        for field_index in xrange(point_defn.GetFieldCount()):
            point_field_name_list.append(
                point_defn.GetFieldDefn(field_index).GetName())
        # this will hold (x,y) coordinates for each point in its iterator order
        point_coord_list = []
        # this maps fieldnames to a list of the values associated with that
        # fieldname in the order that the points are read in and written to
        # `point_coord_list`.
        feature_attributes_fieldname_map = collections.defaultdict(list)
        for point_feature in point_layer:
            sample_point_geometry = point_feature.GetGeometryRef()
            for field_name in point_field_name_list:
                feature_attributes_fieldname_map[field_name].append(
                    point_feature.GetField(field_name))
            point_coord_list.append(
                (sample_point_geometry.GetX(), sample_point_geometry.GetY()))
        del point_layer
        del point_vector

        # each element will hold the point samples for each raster in the
        # order of `point_coord_list`
        sampled_precip_data_list = []
        for field_name in point_field_name_list:
            sampled_precip_data_list.append(
                pandas.Series(
                    data=feature_attributes_fieldname_map[field_name],
                    name=field_name))
        for basename in sorted(raster_path_id_map.iterkeys()):
            path = raster_path_id_map[basename]
            raster = gdal.Open(path)
            band = raster.GetRasterBand(1)
            geotransform = raster.GetGeoTransform()
            sample_list = []
            for point_x, point_y in point_coord_list:
                raster_x = int((
                    point_x - geotransform[0]) / geotransform[1])
                raster_y = int((
                    point_y - geotransform[3]) / geotransform[5])
                sample_list.append(
                    band.ReadAsArray(raster_x, raster_y, 1, 1)[0, 0])
            sampled_precip_data_list.append(
                pandas.Series(data=sample_list, name=basename))

        report_table = pandas.DataFrame(data=sampled_precip_data_list)
        report_table = report_table.transpose()
        report_table.to_csv(save_as)


    def write_wth_files(
            soil_table, temperature_table, precip_table, save_dir):
        """Generate .wth files from temperature and precip tables."""
        temperature_df = pandas.read_csv(temperature_table)
        prec_df = pandas.read_csv(precip_table).set_index("site_id")
        starting_month = int(aligned_args['starting_month'])
        starting_year = int(aligned_args['starting_year'])

        year_list = list()
        for month_index in xrange(int(aligned_args['n_months'])):
            year_list.append(
                starting_year + (starting_month + month_index - 1) // 12)
        year_list = set(year_list)

        site_df = pandas.read_csv(soil_table)
        site_list = site_df.site_id.unique().tolist()
        precip_basename = os.path.basename(
            aligned_args['monthly_precip_path_pattern'])
        for site in site_list:
            temp_subs = temperature_df.loc[temperature_df['site'] == site]
            temp_subs = temp_subs.set_index('month')
            trans_dict = {m: [] for m in range(1, 13)}
            trans_dict['label'] = []
            trans_dict['year'] = []
            for year in year_list:
                trans_dict['year'].append(int(year))
                trans_dict['label'].append('prec')
                for mon in range(1, 13):
                    column = precip_basename.replace(
                                    '<year>', str(year)).replace(
                                    '<month>', '%.2d' % mon)
                    try:
                        prec = prec_df.loc[site, column]
                    except KeyError:
                        prec = 0.
                    trans_dict[mon].append(prec)
                trans_dict['year'].append(int(year))
                trans_dict['label'].append('tmin')
                for mon in range(1, 13):
                    tmin = temp_subs.get_value(mon, 'tmin')
                    trans_dict[mon].append(tmin)
                trans_dict['year'].append(int(year))
                trans_dict['label'].append('tmax')
                for mon in range(1, 13):
                    tmax = temp_subs.get_value(mon, 'tmax')
                    trans_dict[mon].append(tmax)
            df = pandas.DataFrame(trans_dict)
            cols = df.columns.tolist()
            cols = cols[-2:-1] + cols[-1:] + cols[:-2]
            df = df[cols]
            df['sort_col'] = df['year']
            df.loc[(df['label'] == 'prec'), 'sort_col'] = df.sort_col + 0.1
            df.loc[(df['label'] == 'tmin'), 'sort_col'] = df.sort_col + 0.2
            df.loc[(df['label'] == 'tmax'), 'sort_col'] = df.sort_col + 0.3
            df = df.sort_values(by='sort_col')
            df = df.drop('sort_col', 1)
            formats = ['%4s', '%6s'] + ['%7.2f'] * 12
            save_as = os.path.join(save_dir, '{}.wth'.format(site))
            numpy.savetxt(save_as, df.values, fmt=formats, delimiter='')


    def write_site_files(
            aligned_args, soil_table, worldclim_precip_table,
            temperature_table, inputs_dir):
        """Write the site.100 file for each point simulation.

        Use the template_100 as a template and copy other site characteristics
        from the soil table.  Take average climate inputs from Worldclim
        temperature and precipitation tables.
        """
        prec_df = pandas.read_csv(worldclim_precip_table)
        temp_df = pandas.read_csv(temperature_table)

        bulkd_key = os.path.basename(
            aligned_args['bulk_density_path'])[:-4][:10]
        ph_key = os.path.basename(aligned_args['ph_path'])[:-4][:10]
        clay_key = os.path.basename(
            aligned_args['clay_proportion_path'])[:-4][:10]
        silt_key = os.path.basename(
            aligned_args['silt_proportion_path'])[:-4][:10]
        sand_key = os.path.basename(
            aligned_args['sand_proportion_path'])[:-4][:10]

        in_list = pandas.read_csv(soil_table).to_dict(orient="records")
        for inputs_dict in in_list:
            prec_subs = prec_df.loc[prec_df['site'] == inputs_dict['site_id']]
            prec_subs = prec_subs.set_index('month')
            temp_subs = temp_df.loc[temp_df['site'] == inputs_dict['site_id']]
            temp_subs = temp_subs.set_index('month')
            fh, abs_path = mkstemp()
            os.close(fh)
            with open(abs_path, 'w') as newfile:
                first_line = (
                    '%d (generated by script)\r\n' %
                    int(inputs_dict['site_id']))
                newfile.write(first_line)
                with open(TEMPLATE_100, 'r') as old_file:
                    next(old_file)
                    line = old_file.readline()
                    while 'PRECIP(1)' not in line:
                        newfile.write(line)
                        line = old_file.readline()
                    for m in range(1, 13):
                        item = prec_subs.get_value(m, 'prec')
                        newline = '{:<8.5f}          \'PRECIP({})\'\n'.format(
                                        item, m)
                        newfile.write(newline)
                    for m in range(1, 13):
                        newfile.write(
                            '0.00000           \'PRCSTD({})\'\n'.format(m))
                    for m in range(1, 13):
                        newfile.write(
                            '0.00000           \'PRCSKW({})\'\n'.format(m))
                    while 'TMN2M(1)' not in line:
                        line = old_file.readline()
                    for m in range(1, 13):
                        item = temp_subs.get_value(m, 'tmin')
                        newline = '{:<8.5f}          \'TMN2M({})\'\n'.format(
                                    item, m)
                        newfile.write(newline)
                    for m in range(1, 13):
                        item = temp_subs.get_value(m, 'tmax')
                        newline = '{:<8.5f}          \'TMX2M({})\'\n'.format(
                                    item, m)
                        newfile.write(newline)
                    while 'TM' in line:
                        line = old_file.readline()
                    while line:
                        if '  \'SITLAT' in line:
                            item = '{:0.12f}'.format(
                                        inputs_dict['latitude'])[:7]
                            newline = '%s           \'SITLAT\'\r\n' % item
                        elif '  \'SITLNG' in line:
                            item = '{:0.12f}'.format(
                                        inputs_dict['longitude'])[:7]
                            newline = '%s           \'SITLNG\'\r\n' % item
                        elif '  \'SAND' in line:
                            num = inputs_dict[sand_key]
                            item = '{:0.12f}'.format(num)[:7]
                            newline = '%s           \'SAND\'\r\n' % item
                        elif '  \'SILT' in line:
                            num = inputs_dict[silt_key]
                            item = '{:0.12f}'.format(num)[:7]
                            newline = '%s           \'SILT\'\r\n' % item
                        elif '  \'CLAY' in line:
                            num = inputs_dict[clay_key]
                            item = '{:0.12f}'.format(num)[:7]
                            newline = '%s           \'CLAY\'\r\n' % item
                        elif '  \'BULKD' in line:
                            item = '{:0.12f}'.format(inputs_dict[
                                           bulkd_key])[:7]
                            newline = '%s           \'BULKD\'\r\n' % item
                        elif '  \'PH' in line:
                            item = '{:0.12f}'.format(inputs_dict[
                                           ph_key])[:7]
                            newline = '%s           \'PH\'\r\n' % item
                        else:
                            newline = line
                        newfile.write(newline)
                        try:
                            line = old_file.readline()
                        except StopIteration:
                            break
            save_as = os.path.join(
                inputs_dir, '{}.100'.format(int(inputs_dict['site_id'])))
            shutil.copyfile(abs_path, save_as)
            os.remove(abs_path)


    def write_sch_files(soil_table, save_dir):
        """Write the schedule file for each point simulation.

        Use the template site and schedule files and copy other site
        characteristics from the soil table.
        """
        def copy_sch_file(template, site_name, save_as, wth_file=None):
            fh, abs_path = mkstemp()
            os.close(fh)
            with open(abs_path, 'w') as newfile:
                with open(template, 'r') as old_file:
                    for line in old_file:
                        if '  Weather choice' in line:
                            if wth_file:
                                newfile.write(
                                    'F             Weather choice\r\n')
                                newfile.write('{}\r\n'.format(wth_file))
                                line = old_file.readline()
                            else:
                                newfile.write(
                                    'M             Weather choice\r\n')
                                line = old_file.readline()
                        if '.wth\r\n' in line:
                            line = old_file.readline()
                        if '  Site file name' in line:
                            item = '{:14}'.format('{}.100'.format(site_name))
                            newfile.write('{}Site file name\r\n'.format(item))
                        else:
                            newfile.write(line)

            shutil.copyfile(abs_path, save_as)
            os.remove(abs_path)
        site_df = pandas.read_csv(soil_table)
        site_list = site_df.site_id.unique().tolist()
        for site in site_list:
            save_as = os.path.join(save_dir, '{}.sch'.format(site))
            wth = '{}.wth'.format(site)
            copy_sch_file(TEMPLATE_SCH, site, save_as, wth_file=wth)
            save_as = os.path.join(save_dir, '{}_hist.sch'.format(site))
            copy_sch_file(TEMPLATE_HIST, site, save_as)

    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)
    aligned_args = generate_aligned_inputs()
    soil_table = aligned_args['site_table']
    precip_table = os.path.join(processing_dir, "chirps_precip.csv")
    worldclim_precip_table = os.path.join(
        processing_dir, "worldclim_precip.csv")
    temperature_table = os.path.join(
        processing_dir, "temperature_table.csv")

    # generate GRID_POINT_SHP from one of the aligned inputs, using the aligned
    # inputs as template raster. GRID_POINT_SHP is used to index raster pixels
    # to individual points where we launch old Century for each point, then
    # reassemble into a raster later on
    generate_site_shp_from_raster(
        aligned_args['site_param_spatial_index_path'], GRID_POINT_SHP)

    # make point-based inputs from rasters, drawing values from pixels
    # intersecting each point in GRID_POINT_SHP
    write_soil_table(GRID_POINT_SHP, soil_table)
    write_temperature_table(GRID_POINT_SHP, temperature_table)

    write_precip_table_from_rasters(aligned_args, GRID_POINT_SHP, precip_table)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    write_wth_files(
        soil_table, temperature_table, precip_table, input_dir)
    write_worldclim_precip_table(GRID_POINT_SHP, worldclim_precip_table)
    write_site_files(
        aligned_args, soil_table, worldclim_precip_table, temperature_table,
        input_dir)
    write_sch_files(soil_table, input_dir)


def launch_old_model(old_model_input_dir, old_model_output_dir):
    """Run the old model for a series of points."""
    def modify_stocking_density(herbivore_csv, new_sd):
        """Modify the stocking density in the herbivore csv."""
        df = pandas.read_csv(herbivore_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one herbivore type"
        df['stocking_density'] = df['stocking_density'].astype(float)
        df.set_value(0, 'stocking_density', new_sd)
        df.to_csv(herbivore_csv)

    def edit_grass_csv(grass_csv, label):
        """Edit the grass csv to reflect a new label.

        The new label which points to the Century inputs. This allows us to use
        one grass csv for multiple sites. Century inputs must exist.
        """
        df = pandas.read_csv(grass_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one grass type"
        df.set_value(0, 'label', label)
        df[['label']] = df[['label']].astype(type(label))
        df.to_csv(grass_csv)

    input_args = generate_aligned_inputs()
    old_model_args = {
            'input_dir': old_model_input_dir,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 106.4,  # 1,
            'start_year': input_args['starting_year'],
            'start_month': input_args['starting_month'],
            'num_months': input_args['n_months'],
            'mgmt_threshold': 0.1,  # 300.,
            'century_dir': CENTURY_DIR,
            'template_level': input_args['template_level'],
            'fix_file': input_args['fix_file'],
            'user_define_protein': 0,
            'user_define_digestibility': 0,
            'herbivore_csv': input_args['herbivore_csv'],
            'grass_csv': input_args['grass_csv'],
            'digestibility_flag': 'CPER',
            }

    modify_stocking_density(old_model_args['herbivore_csv'], 0.1)  # TODO relate sd to new model
    site_list = pandas.read_csv(
        input_args['site_table']).to_dict(orient='records')
    outer_outdir = old_model_output_dir
    for site in site_list:
        old_model_args['latitude'] = site['latitude']
        # TODO remove me
        old_model_args['latitude'] = 45
        old_model_args['outdir'] = os.path.join(
            outer_outdir, '{}'.format(int(site['site_id'])))
        if not os.path.isfile(os.path.join(
                old_model_args['outdir'], 'summary_results.csv')):
            edit_grass_csv(old_model_args['grass_csv'], int(site['site_id']))
            # try:
            old_model.execute(old_model_args)
            # except:
                # import pdb; pdb.set_trace()
                # continue


def table_to_raster(
        table, template_raster_path, field_list, grid_point_shp, save_dir,
        save_as_field_list=None):
    """Generate a series of rasters from values in a table.

    One raster is generated for each of the fields `field_list` in `table`.
    First join the values from the table to a shapefile containing points,
    then generate a raster for each field in `field_list` from the points
    shapefile. `table` must contain a field `site_id` that can be matched to
    `site_id` in the shapefile `grid_point_shp`.
    If `save_as_field_list` is supplied, it must contain the basename for each
    raster to be saved, in the same order as `field_list`.
    """
    cell_size = pygeoprocessing.get_raster_info(
        template_raster_path)['pixel_size'][0]
    tempdir = tempfile.mkdtemp()
    arcpy.env.workspace = tempdir
    arcpy.env.overwriteOutput = True

    table_df = pandas.read_csv(table)
    # table_df = table_df[['site_id'] + field_list].set_index('site_id')
    table_df = table_df.set_index('site_id')

    # replace all instances of "," with "_" in column headers
    rename_dict = {col: re.sub(',', '_', col) for col in table_df.columns}
    table_df.rename(columns=rename_dict)
    temp_table_out = os.path.join(tempdir, 'temp_table.csv')
    table_df.to_csv(temp_table_out)
    shp_fields = [f.name for f in arcpy.ListFields(grid_point_shp)]
    assert (
        len(set(shp_fields).intersection(set(
            table_df.reset_index().columns.values))) == 1,
        "Table and shapefile must contain 1 matching value")
    # use this key to keep track of field names in original data sources,
    # since the join and export will change them
    if save_as_field_list:
        field_key = shp_fields + ['site_id'] + save_as_field_list
    else:
        field_key = shp_fields + ['site_id'] + field_list

    arcpy.MakeFeatureLayer_management(grid_point_shp, 'temp_layer')
    arcpy.MakeTableView_management(temp_table_out, 'temp_table')
    arcpy.AddJoin_management('temp_layer', 'site_id', 'temp_table', 'site_id')
    temp_shp_out = os.path.join(tempdir, 'points.shp')
    arcpy.CopyFeatures_management('temp_layer', temp_shp_out)
    joined_field_list = [f.name for f in arcpy.ListFields(temp_shp_out)]
    assert (
        len(shp_fields) + len(field_list) + 1 == len(joined_field_list),
        "This is not working as expected")
    start_idx = len(joined_field_list) - len(field_list)
    end_idx = len(joined_field_list)
    for field_idx in xrange(start_idx, end_idx):
        field_name = field_key[field_idx]
        shp_field = joined_field_list[field_idx]
        raster_path = os.path.join(save_dir, '{}.tif'.format(field_name))
        arcpy.PointToRaster_conversion(
            temp_shp_out, shp_field, raster_path, cellsize=cell_size)
    check_raster_dimensions(raster_path, template_raster_path)


def old_model_results_to_table(
        old_model_output_dir, results_table_path, output_list, start_time,
        end_time):
    """Collect results from old model.

    Results from old model may be used as initial values or targets for
    regression testing of new model.  Century outputs to collect should be
    supplied in `output_list`. Outputs will be collected that fall in
    [start_time, end_time].
    """
    input_args = generate_aligned_inputs()
    output_list.insert(0, 'time')

    # collect from raw Century outputs, the last month of the simulation
    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = int(input_args['n_months']) - 1
    last_month = (starting_month + month_i - 1) % 12 + 1
    last_year = starting_year + (starting_month + month_i - 1) // 12
    century_out_basename = 'CENTURY_outputs_m{:d}_y{:d}'.format(
                                                        last_month, last_year)
    site_df = pandas.read_csv(input_args['site_table'])
    df_list = []
    failed_sites = []
    for site in site_df['site_id']:
        century_out_file = os.path.join(old_model_output_dir,
                                        '{}'.format(int(site)),
                                        century_out_basename,
                                        '{}.lis'.format(int(site)))
        try:
            cent_df = pandas.read_fwf(century_out_file, skiprows=[1])
        except IOError:
            failed_sites.append(site)
            continue
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
                century_out_file, skiprows=[1], widths=widths)
            # mistakes in Century writing results
            if 'minerl(10,1' in cent_df.columns.values:
                cent_df.rename(
                    index=str, columns={'minerl(10,1': 'minerl(10,1)'},
                    inplace=True)
            if 'minerl(10,2' in cent_df.columns.values:
                cent_df.rename(
                    index=str, columns={'minerl(10,2': 'minerl(10,2)'},
                    inplace=True)
        df_subset = cent_df[(cent_df.time >= start_time) &
                            (cent_df.time <= end_time)]
        df_subset = df_subset.drop_duplicates('time')
        outputs = df_subset[output_list]
        outputs = outputs.loc[:, ~outputs.columns.duplicated()]
        outputs['site_id'] = site
        df_list.append(outputs)
    century_results = pandas.concat(df_list)

    # reshape to "wide" format
    century_reshape = pandas.pivot_table(
        century_results, values=output_list[1:], index='site_id',
        columns='time')
    cols = [
        '{}_{}_{}'.format(
            c[0], convert_to_year_month(c[1])[0],
            convert_to_year_month(c[1])[1]) for c in
        century_reshape.columns.values]
    century_reshape.columns = cols
    century_reshape.to_csv(results_table_path)
    print("The following sites failed:")
    print(failed_sites)


def century_params_to_new_model_params(pft_param_path, site_param_path):
    """Generate parameter inputs for the new forage model.

    Site and pft parameters for the new forage model come from various
    files used by Century.  Gather these parameters together from all
    Century parameter files and format them as csvs as expected by new
    forage model.
    """
    new_model_args = generate_base_args()
    # parameter table containing only necessary parameters
    parameter_table = pandas.read_csv(
        os.path.join(
            DROPBOX_DIR,
            "Forage_model/CENTURY4.6/GK_doc/Century_parameter_table.csv"))
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
    with open(os.path.join(CENTURY_DIR, new_model_args['fix_file']),
              'r') as siteparam:
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

    # add to grass csv to make PFT trait table
    PFT_param_dict['growth_months'] = (
        [','.join([str(m) for m in range(
            int(list(first_month)[0]), int(list(last_month)[0]) + 1)])])
    if senescence_month:
        PFT_param_dict['senescence_month'] = (
            ','.join([str(m) for m in list(senescence_month)]))
    grass_df = pandas.read_csv(new_model_args['grass_csv'])
    if grass_df.type[0] == 'C3':
        PFT_param_dict['species_factor'] = 0
    else:
        PFT_param_dict['species_factor'] = 0.16
    pft_df = pandas.DataFrame(PFT_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in pft_df.columns.values}
    pft_df.rename(index=int, columns=col_rename_dict, inplace=True)
    pft_df.to_csv(pft_param_path, index=False)
    # pft_df.to_csv(new_model_args['veg_trait_path'], index=False)

    # make site parameter table
    site_df = pandas.DataFrame(site_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in site_df.columns.values}
    site_df.rename(index=int, columns=col_rename_dict, inplace=True)
    site_df.to_csv(site_param_path, index=False)
    # site_df.to_csv(new_model_args['site_param_path'], index=False)

    # TODO: get animal parameters from new_model_args['herbivore_csv']
    # TODO: add new PFT parameters:
    #   digestibility_slope
    #   digestibility_intercept


def initial_variables_to_outvars():
    """Make new outvars file from variables needed to initialize new model.

    The outvars file contains output variables collected by Century.
    """
    initial_var_table = os.path.join(
        DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
    outvars_table = os.path.join(CENTURY_DIR, "outvars.txt")

    init_var_df = pandas.read_csv(initial_var_table)
    init_var_df['outvars'] = [
        v.lower() for v in init_var_df.State_variable_Century]
    outvar_df = init_var_df['outvars']
    outvar_df.to_csv(outvars_table, header=False, index=False, sep='\t')


def generate_initialization_rasters():
    """Run the old model and collect results for intialization.

    The new model may be initialized with rasters for each state variable.
    Collect these rasters from results of a run of the old model.
    """
    initialization_dir = os.path.join(SAMPLE_DATA, "initialization_data")
    if not os.path.exists(initialization_dir):
        os.makedirs(initialization_dir)
    input_args = generate_aligned_inputs()

    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = -1
    target_month = (starting_month + month_i - 1) % 12 + 1
    target_year = starting_year + (starting_month + month_i - 1) // 12
    start_time = convert_to_century_date(target_year, target_month)
    end_time = start_time

    # initial_variables_to_outvars()
    # TODO remove me
    old_model_output_dir = "C:/Users/ginge/Desktop"
    launch_old_model(old_model_input_dir, old_model_output_dir)

    # intialization rasters
    outvar_csv = os.path.join(
        DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
    outvar_df = pandas.read_csv(outvar_csv)
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    outvar_df.sort_values(by=['outvar'], inplace=True)
    for sbstr in ['PFT', 'site']:
        output_list = outvar_df[
            outvar_df.Property_of == sbstr].outvar.tolist()
        field_list = ['{}_{}_{:02d}'.format(
            f, target_year, target_month) for f in output_list]
        results_table_path = os.path.join(
            old_model_output_dir, '{}_initial.csv'.format(sbstr))
        old_model_results_to_table(
            old_model_output_dir, results_table_path, output_list, start_time,
            end_time)
        save_as_field_list = (
            outvar_df[outvar_df.Property_of == sbstr].
            State_variable_rangeland_production.tolist())
        if sbstr == 'PFT':
            # hack that assumes we have just one PFT
            save_as_field_list = ['{}_1'.format(f) for f in save_as_field_list]
        assert len(save_as_field_list) == len(field_list), """Save as field
            list must be of equal length to field list"""
        save_dir = initialization_dir
        table_to_raster(
            results_table_path, input_args['site_param_spatial_index_path'],
            field_list, GRID_POINT_SHP, save_dir,
            save_as_field_list=save_as_field_list)


def generate_regression_tests(regression_testing_dir):
    """Collect regression results from a run of the old model."""
    if not os.path.exists(regression_testing_dir):
        os.makedirs(regression_testing_dir)
    input_args = generate_aligned_inputs()

    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = int(input_args['n_months']) - 1
    end_month = (starting_month + month_i - 1) % 12 + 1
    end_year = starting_year + (starting_month + month_i - 1) // 12
    start_time = convert_to_century_date(end_year, end_month)
    end_time = start_time

    initial_variables_to_outvars()
    launch_old_model(old_model_input_dir, old_model_output_dir)

    # regression testing rasters
    outvar_df = pandas.read_csv(
        os.path.join(
            DROPBOX_DIR,
            "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv"))
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    outvar_df.sort_values(by=['outvar'], inplace=True)
    for sbstr in ['site', 'PFT']:
        output_list = outvar_df[
            outvar_df.Property_of == sbstr].outvar.tolist()
        field_list = ['{}_{}_{:d}'.format(
            f, end_year, end_month) for f in output_list]
        results_table_path = os.path.join(
            old_model_output_dir, '{}_regression_test.csv'.format(sbstr))
        old_model_results_to_table(
            old_model_output_dir, results_table_path, output_list, start_time,
            end_time)
        save_as_field_list = (
            outvar_df[outvar_df.Property_of ==
            sbstr].State_variable_rangeland_production.tolist())
        if sbstr == 'PFT':
            # hack that assumes we have just one PFT
            save_as_field_list = ['{}_1'.format(f) for f in save_as_field_list]
        assert len(save_as_field_list) == len(field_list), """Save as field
            list must be of equal length to field list"""
        save_dir = regression_testing_dir
        table_to_raster(
            results_table_path, input_args['site_param_spatial_index_path'],
            field_list, GRID_POINT_SHP, save_dir,
            save_as_field_list=save_as_field_list)


def generate_biomass_rasters(biomass_raster_dir):
    """Generate live biomass rasters from the old model.

    Generate a raster time series of aboveground live biomass from a model run
    at a series of points emulating a raster.  One raster is generated from
    each monthly time step of the model run.  This function launches the model
    to generate results for each pixel in a series of raster inputs. After the
    aboveground biomass rasters are generated from model results, this function
    deletes intermediate files (i.e., Century raw output files) from disk.

    Returns:
        None
    """
    if not os.path.exists(biomass_raster_dir):
        os.makedirs(biomass_raster_dir)
    input_args = generate_aligned_inputs()

    # launch_old_model(old_model_input_dir, old_model_output_dir)
    erase_intermediate_files(old_model_output_dir)

    site_index_list = [
        f for f in os.listdir(old_model_output_dir) if os.path.isdir(
            os.path.join(old_model_output_dir, f))]
    df_list = []
    for site_id in site_index_list:
        summary_csv = os.path.join(
            old_model_output_dir, site_id, 'summary_results.csv')
        try:
            site_df = pandas.read_csv(summary_csv)
        except IOError:
            continue
        site_df = (
            site_df[site_df['step'] >= 0][[
            '{}_green_kgha'.format(site_id), 'month', 'year']])
        site_df['site_id'] = site_id
        site_df['month_year'] = site_df["month"].map(str) + "_" + site_df["year"].map(str)
        # reshape to "wide" format
        reshape_df = pandas.pivot_table(
            site_df, values='{}_green_kgha'.format(site_id), index='site_id',
            columns='month_year')
        df_list.append(reshape_df)
    old_model_results = pandas.concat(df_list)
    results_table_path = os.path.join(old_model_output_dir, 'green_biomass_time_series.csv')
    old_model_results.to_csv(results_table_path)

    field_list = old_model_results.columns.values
    save_as_field_list = ['live_biomass_kgha_{}'.format(f) for f in field_list]
    table_to_raster(
        results_table_path, input_args['site_param_spatial_index_path'],
        field_list, GRID_POINT_SHP, biomass_raster_dir, save_as_field_list=save_as_field_list)


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


def check_nodata_values():
    """Reclassify nodata values of initialization rasters to match each other.

    Check nodata values of initialization rasters, and if there is more than
    one value, reclassify the nodata value of the offending raster so that it
    matches the others.

    """
    initialization_dir = r"C:\Users\ginge\Dropbox\sample_inputs\initialization_data"
    wrong_list = []
    basename_list = [
        f for f in os.listdir(initialization_dir) if f.endswith('.tif')]
    state_var_nodata = pygeoprocessing.get_raster_info(
        os.path.join(initialization_dir, basename_list[0]))['nodata'][0]
    for bn in basename_list[1:]:
        nodata = pygeoprocessing.get_raster_info(
            os.path.join(initialization_dir, bn))['nodata'][0]
        if nodata != state_var_nodata:
            raster_path = os.path.join(initialization_dir, bn)
            raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(1)
            raster_band.SetNoDataValue(state_var_nodata)
            raster_band.FlushCache()
            raster.FlushCache()
            raster_band = None
            raster = None
        nodata = pygeoprocessing.get_raster_info(
            os.path.join(initialization_dir, bn))['nodata'][0]
        if nodata != state_var_nodata:
            wrong_list.append(bn)
    print(wrong_list)


if __name__ == "__main__":
    # sample results for Lingling, 5x5 CHIRPS pixels
    old_model_processing_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_inputs/Manlai_soum_WGS84")
    old_model_input_dir = os.path.join(
        old_model_processing_dir, 'model_inputs')
    old_model_output_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_results/Manlai_soum_WGS84")
    regression_testing_dir = os.path.join(
        old_model_output_dir, 'regression_test_data')
    biomass_raster_dir = os.path.join(
        old_model_output_dir, 'biomass_rasters')

    # directories for model testing results
    old_model_processing_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_inputs/pycentury_dev")
    old_model_input_dir = os.path.join(
        old_model_processing_dir, 'model_inputs')
    old_model_output_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_results/pycentury_dev")
    regression_testing_dir = "C:/Users/ginge/Documents/NatCap/regression_test_data"

    # site and pft inputs derived from calibrated Century inputs
    pft_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/pft_trait.csv"
    site_param_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/site_parameters.csv"
    century_params_to_new_model_params(pft_param_path, site_param_path)
    # TODO change n_months in base args to 12
    # generate_inputs_for_old_model(
    #     old_model_processing_dir, old_model_input_dir)
    # generate_initialization_rasters()
    # TODO change n_months in base args to 1
    # generate_regression_tests(regression_testing_dir)
    # generate_biomass_rasters(biomass_raster_dir)
    # generate_aligned_inputs()
    # check_nodata_values()

    # century_output_file = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/pycentury_dev/0/CENTURY_outputs_m12_y2016/0.lis"
    # year = 2015
    # month = 12
    # site_initial_table = "C:/Users/ginge/Dropbox/sample_inputs/site_initial_table.csv"
    # pft_initial_table = "C:/Users/ginge/Dropbox/sample_inputs/pft_initial_table.csv"
    # century_outputs_to_initial_tables(
    #     century_output_file, year, month, site_initial_table,
    #     pft_initial_table)
