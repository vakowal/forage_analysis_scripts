## Process raw data, generate inputs for old forage model, generate inputs
## for new forage model (PyCentury), and launch old forage model to generate
## regression testing results to test new forage model against.

import os
import sys
import pandas as pd
import glob
import numpy as np
from osgeo import ogr
from osgeo import gdal
import collections
import taskgraph
import arcpy
import tempfile
import shutil
import re
import math
from tempfile import mkstemp

sys.path.append("C:/Users/ginge/Documents/Python/rangeland_production")
import forage as old_model


arcpy.CheckOutExtension("Spatial")

SAMPLE_DATA = "C:/Users/ginge/Documents/NatCap/sample_inputs"
TEMPLATE_100 = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\no_grazing.100"
TEMPLATE_HIST = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\no_grazing_GCD_G1_hist.sch"
TEMPLATE_SCH = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\no_grazing_GCD_G1.sch"
CENTURY_DIR = 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
DROPBOX_DIR = "C:/Users/ginge/Dropbox/NatCap_backup"
LOCAL_DIR = "E:/GIS_local_8.27.18"  # "C:/Users/ginge/Documents/NatCap/GIS_local"


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


def generate_base_args():
    """These raw inputs should work for both old and new models."""
    args = {
            'starting_month': 1,
            'starting_year': 2016,
            'n_months': 1,
            'aoi_path': os.path.join(
                SAMPLE_DATA, 'Manlai_soum.shp'),
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
            'site_param_raster_path_pattern': os.path.join(
                SAMPLE_DATA, 'site<site>.tif'),
            'veg_trait_path': os.path.join(SAMPLE_DATA, 'pft_trait.csv'),
            'veg_spatial_composition_path_pattern': os.path.join(
                SAMPLE_DATA, 'pft<PFT>.tif'),
            'animal_trait_path': os.path.join(
                SAMPLE_DATA, 'animal_trait_table.csv'),
            'animal_mgmt_layer_path': os.path.join(
                SAMPLE_DATA, 'sheep_units_density_2016_monitoring_area.shp'),
        }
    return args


def generate_inputs_for_old_model(processing_dir, input_dir):
    """Generate inputs for the old forage model that used the Century
    executable. Most of this taken from 'mongolia_workflow()' in the script
    process_regional_inputs.py. Template site and schedule files should be
    the same as those used to generate inputs for new model.
    Returns a dictionary of inputs that can be used to launch the old model
    for many sites."""

    input_args = generate_base_args()
    point_shp = os.path.join(
        LOCAL_DIR, "Mongolia/CHIRPS/CHIRPS_pixel_centroid_1_soum.shp")

    def write_soil_table(site_shp, save_as):
        """Make soil table to use as input for site files"""

        # make a temporary copy of the point shapefile to append
        # worldclim values
        tempdir = tempfile.mkdtemp()
        source_shp = site_shp
        site_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, site_shp)

        raster_files = [input_args['bulk_density_path'],
                        input_args['ph_path'],
                        input_args['clay_proportion_path'],
                        input_args['silt_proportion_path'],
                        input_args['sand_proportion_path']]
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
        soil_df = pd.DataFrame.from_dict(temp_dict).set_index('site_id')

        bulkd_key = os.path.basename(input_args['bulk_density_path'])[:-4][:10]
        ph_key = os.path.basename(input_args['ph_path'])[:-4][:10]
        clay_key = os.path.basename(
            input_args['clay_proportion_path'])[:-4][:10]
        silt_key = os.path.basename(
            input_args['silt_proportion_path'])[:-4][:10]
        sand_key = os.path.basename(
            input_args['sand_proportion_path'])[:-4][:10]
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
        """Make a table of max and min monthly temperature for points where
        we will launch the old model."""

        tempdir = tempfile.mkdtemp()
        # make a temporary copy of the point shapefile to append
        # worldclim values
        source_shp = site_shp
        point_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, point_shp)

        temperature_month_set = set()
        starting_month = int(input_args['starting_month'])
        for month_index in xrange(int(input_args['n_months'])):
            month_i = (starting_month + month_index - 1) % 12 + 1
            temperature_month_set.add(month_i)

        raster_files = []
        field_list = []
        for substring in ['min', 'max']:
            for month_i in temperature_month_set:
                monthly_temp_path = input_args[
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
                temp_dict['site'].extend([site] * 12)
                for f_idx in range(1, len(field_list)):
                    field = field_list[f_idx]
                    if field.startswith('min'):
                        temp_dict['tmin'].append(row[f_idx])
                    elif field.startswith('max'):
                        temp_dict['tmax'].append(row[f_idx])
                    else:
                        raise Exception, "value not recognized"
                temp_dict['month'].extend(temperature_month_set)
        for key in temp_dict.keys():
            if len(temp_dict[key]) == 0:
                del temp_dict[key]
        temp_df = pd.DataFrame.from_dict(temp_dict)
        # check units
        while max(temp_df['tmax']) > 100:
            temp_df['tmax'] = temp_df['tmax'] / 10.
            temp_df['tmin'] = temp_df['tmin'] / 10.
        temp_df.to_csv(save_as, index=False)

    def write_worldclim_precip_table(site_shp, save_as):
        """Write precipitation table from Worldclim average precipitation, to
        be used for spin-up simulations."""

        worldclim_pattern = os.path.join(
            LOCAL_DIR, "Mongolia/Worldclim/precip/wc2.0_30s_prec_<month>.tif")

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
        prec_df = pd.DataFrame.from_dict(prec_dict)
        # divide raw Worldclim precip by 10
        prec_df['prec'] = prec_df['prec'] / 10.
        prec_df.to_csv(save_as, index=False)

    def write_precip_table_from_rasters(precip_dir, site_shp, save_as):
        """Make a table of precipitation from a series of rasters inside
        precip_dir. This adapted from Rich's script point_precip_fetch.py."""

        task_graph = taskgraph.TaskGraph('taskgraph_cache', 0)
        task_graph.close()
        task_graph.join()

        # index all raster paths by their basename
        raster_path_id_map = {}
        for raster_path in glob.glob(os.path.join(precip_dir, '*.tif')):
            basename = os.path.splitext(os.path.basename(raster_path))[0]
            raster_path_id_map[basename] = raster_path

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
                pd.Series(
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
                pd.Series(data=sample_list, name=basename))

        report_table = pd.DataFrame(data=sampled_precip_data_list)
        report_table = report_table.transpose()
        report_table.to_csv(save_as)

    def write_wth_files(soil_table, temperature_table, precip_table,
                        save_dir):
        """Generate .wth files from temperature and precip tables."""

        temperature_df = pd.read_csv(temperature_table)
        prec_df = pd.read_csv(precip_table).set_index("site_id")
        starting_month = int(input_args['starting_month'])
        starting_year = int(input_args['starting_year'])

        year_list = list()
        for month_index in xrange(int(input_args['n_months'])):
            year_list.append(
                starting_year + (starting_month + month_index - 1) // 12)
        year_list = set(year_list)

        site_df = pd.read_csv(soil_table)
        site_list = site_df.site_id.unique().tolist()
        precip_basename = os.path.basename(
            input_args['monthly_precip_path_pattern'])[:-4]
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
                    except:
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
            df = pd.DataFrame(trans_dict)
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
            np.savetxt(save_as, df.values, fmt=formats, delimiter='')

    def write_site_files(soil_table, worldclim_precip_table,
                         temperature_table, inputs_dir):
        """Write the site.100 file for each point simulation, using the
        template_100 as a template, copying other site characteristics
        from the soil table, and taking average climate inputs from
        Worldclim temperature and precipitation tables."""

        prec_df = pd.read_csv(worldclim_precip_table)
        temp_df = pd.read_csv(temperature_table)

        bulkd_key = os.path.basename(input_args['bulk_density_path'])[:-4][:10]
        ph_key = os.path.basename(input_args['ph_path'])[:-4][:10]
        clay_key = os.path.basename(
            input_args['clay_proportion_path'])[:-4][:10]
        silt_key = os.path.basename(
            input_args['silt_proportion_path'])[:-4][:10]
        sand_key = os.path.basename(
            input_args['sand_proportion_path'])[:-4][:10]

        in_list = pd.read_csv(soil_table).to_dict(orient="records")
        for inputs_dict in in_list:
            prec_subs = prec_df.loc[prec_df['site'] == inputs_dict['site_id']]
            prec_subs = prec_subs.set_index('month')
            temp_subs = temp_df.loc[temp_df['site'] == inputs_dict['site_id']]
            temp_subs = temp_subs.set_index('month')
            fh, abs_path = mkstemp()
            os.close(fh)
            with open(abs_path, 'wb') as newfile:
                first_line = (
                    '%d (generated by script)\r\n' %
                    int(inputs_dict['site_id']))
                newfile.write(first_line)
                with open(TEMPLATE_100, 'rb') as old_file:
                    next(old_file)
                    line = old_file.next()
                    while 'PRECIP(1)' not in line:
                        newfile.write(line)
                        line = old_file.next()
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
                        line = old_file.next()
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
                        line = old_file.next()
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
                            line = old_file.next()
                        except StopIteration:
                            break
            save_as = os.path.join(inputs_dir, '{}.100'.format(
                                                 int(inputs_dict['site_id'])))
            shutil.copyfile(abs_path, save_as)
            os.remove(abs_path)

    def write_sch_files(soil_table, save_dir):
        """Write the schedule file for each point simulation, using the
        template site and schedule files and copying other site characteristics
        from the soil table."""

        def copy_sch_file(template, site_name, save_as, wth_file=None):
            fh, abs_path = mkstemp()
            os.close(fh)
            with open(abs_path, 'wb') as newfile:
                with open(template, 'rb') as old_file:
                    for line in old_file:
                        if '  Weather choice' in line:
                            if wth_file:
                                newfile.write(
                                    'F             Weather choice\r\n')
                                newfile.write('{}\r\n'.format(wth_file))
                                line = old_file.next()
                            else:
                                newfile.write(
                                    'M             Weather choice\r\n')
                                line = old_file.next()
                        if '.wth\r\n' in line:
                            line = old_file.next()
                        if '  Site file name' in line:
                            item = '{:14}'.format('{}.100'.format(site_name))
                            newfile.write('{}Site file name\r\n'.format(item))
                        else:
                            newfile.write(line)

            shutil.copyfile(abs_path, save_as)
            os.remove(abs_path)

        site_df = pd.read_csv(soil_table)
        site_list = site_df.site_id.unique().tolist()
        for site in site_list:
            save_as = os.path.join(save_dir, '{}.sch'.format(site))
            wth = '{}.wth'.format(site)
            copy_sch_file(TEMPLATE_SCH, site, save_as, wth_file=wth)
            save_as = os.path.join(save_dir, '{}_hist.sch'.format(site))
            copy_sch_file(TEMPLATE_HIST, site, save_as)

    precip_dir = os.path.dirname(input_args['monthly_precip_path_pattern'])
    soil_table = os.path.join(processing_dir, "soil_table.csv")
    precip_table = os.path.join(processing_dir, "chirps_precip.csv")
    worldclim_precip_table = os.path.join(
        processing_dir, "worldclim_precip.csv")
    temperature_table = os.path.join(
        processing_dir, "temperature_table.csv")

    # write_soil_table(point_shp, soil_table)
    # write_temperature_table(point_shp, temperature_table)
    # write_precip_table_from_rasters(precip_dir, point_shp, precip_table)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    # write_wth_files(
        # soil_table, temperature_table, precip_table, input_dir)

    # write_worldclim_precip_table(point_shp, worldclim_precip_table)
    # write_site_files(
        # soil_table, worldclim_precip_table, temperature_table, input_dir)
    # write_sch_files(soil_table, input_dir)

    # other inputs I don't know how to store yet
    grass_table = os.path.join(
        DROPBOX_DIR, "Mongolia/model_inputs/grass.csv")
    herb_table = os.path.join(
        DROPBOX_DIR, "Mongolia/model_inputs/cashmere_goats.csv")
    template_level = 'GH'
    fix_file = 'drygfix.100'

    old_model_inputs_dict = {
        'site_table': soil_table,
        'grass_csv': grass_table,
        'herbivore_csv': herb_table,
        'template_level': template_level,
        'fix_file': fix_file,
    }
    return old_model_inputs_dict


def launch_old_model(
        old_model_inputs_dict, old_model_input_dir, old_model_output_dir):
    """Run the old model with sample inputs. Most of this taken from the
    script Mongolia_monitoring_sites_launch_forage.py (i.e., results we
    generated for AGU in 2017)"""

    def modify_stocking_density(herbivore_csv, new_sd):
        """Modify the stocking density in the herbivore csv used as input to
        the forage model."""

        df = pd.read_csv(herbivore_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one herbivore type"
        df['stocking_density'] = df['stocking_density'].astype(float)
        df.set_value(0, 'stocking_density', new_sd)
        df.to_csv(herbivore_csv)

    def edit_grass_csv(grass_csv, label):
        """Edit the grass csv to reflect a new label, which points to the
        Century inputs, so we can use one grass csv for multiple sites
        Century inputs must exist."""

        df = pd.read_csv(grass_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one grass type"
        df.set_value(0, 'label', label)
        df[['label']] = df[['label']].astype(type(label))
        df.to_csv(grass_csv)

    input_args = generate_base_args()
    old_model_args = {
            'input_dir': old_model_input_dir,
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'start_year': input_args['starting_year'],
            'start_month': input_args['starting_month'],
            'num_months': input_args['n_months'],
            'mgmt_threshold': 300.,
            'century_dir': CENTURY_DIR,
            'template_level': old_model_inputs_dict['template_level'],
            'fix_file': old_model_inputs_dict['fix_file'],
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': old_model_inputs_dict['herbivore_csv'],
            'grass_csv': old_model_inputs_dict['grass_csv'],
            }

    modify_stocking_density(old_model_args['herbivore_csv'], 0)  # TODO relate sd to new model
    site_list = pd.read_csv(
        old_model_inputs_dict['site_table']).to_dict(orient='records')
    outer_outdir = old_model_output_dir
    for site in site_list:
        old_model_args['latitude'] = site['latitude']
        old_model_args['outdir'] = os.path.join(
            outer_outdir, '{}'.format(int(site['site_id'])))
        if not os.path.isfile(os.path.join(
                old_model_args['outdir'], 'summary_results.csv')):
            edit_grass_csv(old_model_args['grass_csv'], int(site['site_id']))
            try:
                old_model.execute(old_model_args)
            except:
                continue


def table_to_raster(
        table, field_list, grid_point_shp, save_dir, save_as_field_list=None):
    """Generate a series of rasters from values in the fields `field_list` in
    `table`. First join the values from the table to a shapefile containing
    points, then generate a raster for each field in `field_list` from the
    points shapefile.
    `table` must contain a field `site_id` that can be matched to `site_id`
    in the shapefile `grid_point_shp`.
    If `save_as_field_list` is supplied, it must contain the basename for each
    raster to be saved, in the same order as `field_list`."""

    cell_size = 0.05  # hard coded from CHIRPS
    tempdir = tempfile.mkdtemp()
    arcpy.env.workspace = tempdir
    arcpy.env.overwriteOutput = True

    table_df = pd.read_csv(table)
    table_df = table_df[['site_id'] + field_list].set_index('site_id')
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


def old_model_results_to_table(
        old_model_inputs_dict, old_model_output_dir, results_table_path,
        output_list, start_time, end_time):
    """Collect results from old model to use as initial values or targets for
    regression testing of new model.  Century outputs to collect should be
    supplied in output_list. Outputs will be collected that fall in
    [start_time, end_time]."""

    input_args = generate_base_args()
    output_list.insert(0, 'time')

    # collect from raw Century outputs, the last month of the simulation
    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = int(input_args['n_months']) - 1
    last_month = (starting_month + month_i - 1) % 12 + 1
    last_year = starting_year + (starting_month + month_i - 1) // 12
    century_out_basename = 'CENTURY_outputs_m{:d}_y{:d}'.format(
                                                        last_month, last_year)
    site_df = pd.read_csv(old_model_inputs_dict['site_table'])
    df_list = []
    failed_sites = []
    for site in site_df['site_id']:
        century_out_file = os.path.join(old_model_output_dir,
                                        '{}'.format(int(site)),
                                        century_out_basename,
                                        '{}.lis'.format(int(site)))
        try:
            cent_df = pd.read_fwf(century_out_file, skiprows=[1])
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
            cent_df = pd.read_fwf(
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
    century_results = pd.concat(df_list)

    # reshape to "wide" format
    century_reshape = pd.pivot_table(century_results, values=output_list[1:],
                                     index='site_id', columns='time')
    cols = [
        '{}_{}_{}'.format(
            c[0], convert_to_year_month(c[1])[0],
            convert_to_year_month(c[1])[1]) for c in
        century_reshape.columns.values]
    century_reshape.columns = cols
    century_reshape.to_csv(results_table_path)
    print "The following sites failed:"
    print failed_sites


def generate_inputs_for_new_model(old_model_inputs_dict):
    """Generate inputs for the new forage model that includes plant production
    model written in Python.  This should take the same raw inputs as
    'generate_inputs_for_old_model()'."""

    new_model_args = generate_base_args()
    # parameter table containing only necessary parameters
    parameter_table = pd.read_csv(
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
    with open(TEMPLATE_HIST, 'rb') as hist_sch:
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
    with open(TEMPLATE_SCH, 'rb') as hist_sch:
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
    with open(crop_params, 'rb') as cparam:
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
    with open(graz_file, 'rb') as grazparams:
        for line in grazparams:
            if line.startswith(old_model_inputs_dict['template_level']):
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
    with open(TEMPLATE_100, 'rb') as siteparam:
        for line in siteparam:
            label = re.sub(r"\'", "", line[13:].strip()).lower()
            if label in parameters_to_keep:
                value = float(line[:13].strip())
                master_param_dict[label] = value
    # get fixed parameters from old_model_inputs_dict['fix_file']
    with open(os.path.join(CENTURY_DIR, old_model_inputs_dict['fix_file']),
              'rb') as siteparam:
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
    grass_df = pd.read_csv(old_model_inputs_dict['grass_csv'])
    grass_df = grass_df[['type', 'cprotein_green', 'cprotein_dead']]
    pft_df = pd.DataFrame(PFT_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in pft_df.columns.values}
    pft_df.rename(index=int, columns=col_rename_dict, inplace=True)
    pft_table = pd.concat([grass_df, pft_df], axis=1)
    pft_table.to_csv(new_model_args['veg_trait_path'], index=False)

    # make site parameter table
    site_df = pd.DataFrame(site_param_dict, index=[0])
    col_rename_dict = {c: century_to_rp(c) for c in site_df.columns.values}
    site_df.rename(index=int, columns=col_rename_dict, inplace=True)
    site_df.to_csv(new_model_args['site_param_path'], index=False)

    # TODO: get animal parameters from old_model_inputs_dict['herbivore_csv']


def initial_variables_to_outvars():
    """Make new outvars file (the output variables collected by Century) for
    collection of variables needed to initialize new model."""
    initial_var_table = os.path.join(
        DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
    outvars_table = os.path.join(CENTURY_DIR, "outvars.txt")

    init_var_df = pd.read_csv(initial_var_table)
    init_var_df['outvars'] = [
        v.lower() for v in init_var_df.State_variable_Century]
    outvar_df = init_var_df['outvars']
    outvar_df.to_csv(outvars_table, header=False, index=False, sep='\t')


def generate_initialization_rasters():
    """Run the old model and collect results to be used as intialization
    rasters for new model."""

    initialization_dir = (
        "C:/Users/ginge/Documents/NatCap/sample_inputs/initialization_data")
    if not os.path.exists(initialization_dir):
        os.makedirs(initialization_dir)
    grid_point_shp = os.path.join(
        LOCAL_DIR, "Mongolia/CHIRPS/CHIRPS_pixel_centroid_1_soum.shp")
    input_args = generate_base_args()

    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = -1
    target_month = (starting_month + month_i - 1) % 12 + 1
    target_year = starting_year + (starting_month + month_i - 1) // 12
    start_time = convert_to_century_date(target_year, target_month)
    end_time = start_time

    initial_variables_to_outvars()
    launch_old_model(
        old_model_inputs_dict, old_model_input_dir, old_model_output_dir)

    # intialization rasters
    outvar_csv = os.path.join(
        DROPBOX_DIR,
        "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv")
    outvar_df = pd.read_csv(outvar_csv)
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    for sbstr in ['PFT', 'site']:
        output_list = outvar_df[
            outvar_df.Property_of == sbstr].outvar.tolist()
        field_list = ['{}_{}_{:02d}'.format(
            f, target_year, target_month) for f in output_list]
        results_table_path = os.path.join(
            old_model_output_dir, '{}_initial.csv'.format(sbstr))
        old_model_results_to_table(
            old_model_inputs_dict, old_model_output_dir, results_table_path,
            output_list, start_time, end_time)
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
            results_table_path, field_list, grid_point_shp, save_dir,
            save_as_field_list=save_as_field_list)


def generate_regression_tests():
    """Run the old model and collect results to be used as regression testing
    target rasters for new model."""
    regression_test_dir = (
        "C:/Users/ginge/Documents/NatCap/regression_test_data")
    if not os.path.exists(regression_test_dir):
        os.makedirs(regression_test_dir)
    grid_point_shp = os.path.join(
        LOCAL_DIR, "Mongolia/CHIRPS/CHIRPS_pixel_centroid_1_soum.shp")
    input_args = generate_base_args()

    starting_month = int(input_args['starting_month'])
    starting_year = int(input_args['starting_year'])
    month_i = int(input_args['n_months'])
    end_month = (starting_month + month_i - 1) % 12 + 1
    end_year = starting_year + (starting_month + month_i - 1) // 12
    start_time = convert_to_century_date(end_year, end_month)
    end_time = start_time

    initial_variables_to_outvars()
    launch_old_model(
        old_model_inputs_dict, old_model_input_dir, old_model_output_dir)

    # regression testing rasters
    outvar_df = pd.read_csv(
        os.path.join(
            DROPBOX_DIR,
            "Forage_model/CENTURY4.6/GK_doc/Century_state_variables.csv"))
    outvar_df['outvar'] = [v.lower() for v in outvar_df.State_variable_Century]
    for sbstr in ['site']:  # 'PFT'
        output_list = outvar_df[
            outvar_df.Property_of == sbstr].outvar.tolist()
        field_list = ['{}_{}_{:d}'.format(
            f, end_year, end_month) for f in output_list]
        results_table_path = os.path.join(
            old_model_output_dir, '{}_initial.csv'.format(sbstr))
        old_model_results_to_table(
            old_model_inputs_dict, old_model_output_dir, results_table_path,
            output_list, start_time, end_time)
        save_as_field_list = (
            outvar_df[outvar_df.Property_of ==
            sbstr].State_variable_rangeland_production.tolist())
        if sbstr == 'PFT':
            # hack that assumes we have just one PFT
            save_as_field_list = ['{}_1'.format(f) for f in save_as_field_list]
        assert len(save_as_field_list) == len(field_list), """Save as field
            list must be of equal length to field list"""
        save_dir = regression_test_dir
        table_to_raster(results_table_path, field_list, grid_point_shp,
                        save_dir, save_as_field_list=save_as_field_list)


if __name__ == "__main__":
    old_model_processing_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_inputs/pycentury_dev")
    old_model_input_dir = os.path.join(
        old_model_processing_dir, 'model_inputs')
    old_model_output_dir = os.path.join(
        DROPBOX_DIR, "Mongolia/model_results/pycentury_dev")
    regression_testing_dir = os.path.join(
        old_model_output_dir, 'regression_test_data')
    old_model_inputs_dict = generate_inputs_for_old_model(
        old_model_processing_dir, old_model_input_dir)

    generate_inputs_for_new_model(old_model_inputs_dict)
    generate_initialization_rasters()
    generate_regression_tests()
