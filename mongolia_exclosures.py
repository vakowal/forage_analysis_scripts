"""Run Century at WCS's exclosure sites."""
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
import arcpy

sys.path.append("C:/Users/ginge/Documents/Python/rangeland_production")
import forage

arcpy.CheckOutExtension("Spatial")

SAMPLE_DATA = r"C:\Users\ginge\Dropbox\sample_inputs"
TEMPLATE_100 = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\no_grazing.100"
TEMPLATE_HIST = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\historical_schedule_hist.sch"
TEMPLATE_SCH = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\template_files\historical_schedule_exclosures.sch"
CENTURY_DIR = 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014'
LOCAL_DIR = "C:/Users/ginge/Documents/NatCap/GIS_local"


def write_soil_table(aligned_args, site_shp, save_as):
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


def write_temperature_table(aligned_args, site_shp, save_as):
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
        r"E:\GIS_local_archive\General_useful_data\Worldclim_2.0\worldclim_precip",
        "wc2.0_30s_prec_<month>.tif")

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
        aligned_args, soil_table, temperature_table, precip_table, save_dir):
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
        with open(abs_path, 'wb') as newfile:
            first_line = (
                '%s (generated by script)\r\n' %
                inputs_dict['site_id'])
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
        save_as = os.path.join(
            inputs_dir, '{}.100'.format(inputs_dict['site_id']))
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
    site_df = pandas.read_csv(soil_table)
    site_list = site_df.site_id.unique().tolist()
    for site in site_list:
        save_as = os.path.join(save_dir, '{}.sch'.format(site))
        wth = '{}.wth'.format(site)
        copy_sch_file(TEMPLATE_SCH, site, save_as, wth_file=wth)
        save_as = os.path.join(save_dir, '{}_hist.sch'.format(site))
        copy_sch_file(TEMPLATE_HIST, site, save_as)


def write_namem_sch_files(site_stn_match_table, save_dir):
    """Write schedule files for each point simulation with NAMEM .wth files."""
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
    site_df = pandas.read_csv(site_stn_match_table)
    for site in site_df.site_id.unique().tolist():
        save_as = os.path.join(save_dir, '{}.sch'.format(site))
        soum_ctr = site_df.loc[
            (site_df['site_id'] == site)]['name_en'].values[0]
        wth_file = '{}.wth'.format(soum_ctr)
        copy_sch_file(TEMPLATE_SCH, site, save_as, wth_file=wth_file)
        save_as = os.path.join(save_dir, '{}_hist.sch'.format(site))
        copy_sch_file(TEMPLATE_HIST, site, save_as)


def namem_to_wth(namem_precip, namem_temp, input_folder, wc_temp=None,
                 site_match_csv=None):
    """Write weather files from namem records.  If worldclim temp is included,
    get temperature records from worldclim."""

    temp_df = pandas.read_csv(namem_temp)
    prec_df = pandas.read_csv(namem_precip)
    if wc_temp:
        assert site_match_csv, "Must have site_match_csv if wc_temp is used"
        wc_temp_df = pandas.read_csv(wc_temp)
        match_df = pandas.read_csv(site_match_csv)
        wc_temp_df = pandas.merge(wc_temp_df, match_df, left_on='site',
                              right_on='site_id', how='outer')
        wc_temp_df[['station_name']] = wc_temp_df[['name_en']]
        wc_temp_df['year'] = 2010  # hack
        temp_df = wc_temp_df[['station_name', 'month', 'year', 'tmax',
                              'tmin']]
    namem_df = pandas.merge(prec_df, temp_df, how='outer')
    namem_df[['year', 'month']] = namem_df[['year', 'month']].astype(int)
    year_list = namem_df['year'].unique().tolist()
    year_list = range(min(year_list), max(year_list) + 1)
    for site in namem_df.station_name.unique():
        sub_df = namem_df.loc[(namem_df["station_name"] == site)]
        sub_df = sub_df.sort_values(by=['year', 'month'])
        # replace missing values with average value for each month
        grouped = sub_df.groupby('month')
        for year in year_list:
            for month in range(1, 13):
                test_df = sub_df.loc[(sub_df['month'] == month) &
                                     (sub_df['year'] == year)]
                if len(test_df) == 0:
                    placeholder = pandas.DataFrame({'month': [month],
                                                'year': [year]})
                    sub_df = pandas.concat([sub_df, placeholder])
        for mon in range(1, 13):
            for label in ['tmin', 'tmax', 'precip_mm']:
                fill_val = grouped.get_group(mon).mean()[label]
                sub_df.loc[(sub_df['month'] == mon) &
                           (sub_df[label].isnull()), label] = fill_val
        trans_dict = {'label': ['prec'] * len(year_list), 'year': year_list * 3}
        for mon in range(1, 13):
            prec = sub_df.loc[(sub_df["month"] == mon),
                              "precip_mm"].values.tolist()
            if len(prec) != len(year_list):
                import pdb; pdb.set_trace()
            trans_dict[mon] = [v / 10.0 for v in prec]  # namem vals in mm
            tmin = sub_df.loc[(sub_df['month'] == mon), 'tmin'].values.tolist()
            if len(tmin) != len(year_list):
                import pdb; pdb.set_trace()
            tmax = sub_df.loc[(sub_df['month'] == mon), 'tmax'].values.tolist()
            if len(tmax) != len(year_list):
                import pdb; pdb.set_trace()
            trans_dict[mon].extend(tmin)
            trans_dict[mon].extend(tmax)
        trans_dict['label'].extend(['tmin'] * len(year_list))
        trans_dict['label'].extend(['tmax'] * len(year_list))
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
        save_as = os.path.join(input_folder, '{}.wth'.format(site))
        numpy.savetxt(save_as, df.values, fmt=formats, delimiter='')


def generate_inputs_wcs_exclosures(exclosure_shp, processing_dir, input_dir):
    """Run Century at WCS exclosure sites with CHIRPS and NAMEM precip."""
    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)
    soil_table = os.path.join(processing_dir, 'soil_table.csv')
    chirps_precip_table = os.path.join(processing_dir, "chirps_precip.csv")
    worldclim_precip_table = os.path.join(
        processing_dir, "worldclim_precip.csv")
    temperature_table = os.path.join(
        processing_dir, "temperature_table.csv")

    aligned_args = {
        'starting_month': 1,
        'starting_year': 2008,
        'n_months': 132,  # Jan 2008 to Dec 2018
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
            r"E:\GIS_local\Mongolia\CHIRPS\div_by_10",
            'chirps-v2.0.<year>.<month>.tif'),
        'min_temp_path_pattern': os.path.join(
            r"E:\GIS_local_archive\General_useful_data\Worldclim_2.0\worldclim_tmin",
            'wc2.0_30s_tmin_<month>.tif'),
        'max_temp_path_pattern': os.path.join(
            r"E:\GIS_local_archive\General_useful_data\Worldclim_2.0\worldclim_tmax",
            'wc2.0_30s_tmax_<month>.tif'),
        'template_level': 'GH',
        'fix_file': 'drygfix.100',
    }

    # make point-based inputs from rasters, drawing values from pixels
    # intersecting each point in exclosure_shp
    # write_soil_table(aligned_args, exclosure_shp, soil_table)
    # write_temperature_table(aligned_args, exclosure_shp, temperature_table)
    # write_worldclim_precip_table(exclosure_shp, worldclim_precip_table)
    # write_precip_table_from_rasters(
        # aligned_args, exclosure_shp, chirps_precip_table)

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    # write_wth_files(
    #     aligned_args, soil_table, temperature_table, chirps_precip_table,
    #     input_dir)
    # write_site_files(
    #     aligned_args, soil_table, worldclim_precip_table, temperature_table,
    #     input_dir)
    write_sch_files(soil_table, input_dir)


def div_chirps():
    """Divide CHIRPS rasters by 10."""
    raw_dir = r"E:\GIS_local\Mongolia\CHIRPS\MOVEME"
    target_dir = r"E:\GIS_local\Mongolia\CHIRPS\div_by_10"

    def div_by_10(raw_raster):
        """Divide the raw raster by 10."""
        result = numpy.empty(raw_raster.shape, dtype=numpy.float32)
        result[:] = target_nodata
        result[raw_raster != target_nodata] = (
            raw_raster[raw_raster != target_nodata] / 10.)
        return result

    raster_basename_list = [
        f for f in os.listdir(raw_dir) if f.endswith('.tif')]
    target_nodata = pygeoprocessing.get_raster_info(
        os.path.join(raw_dir, raster_basename_list[0]))['nodata'][0]
    for basename in raster_basename_list:
        raw_raster = os.path.join(raw_dir, basename)
        target_raster = os.path.join(target_dir, basename)
        pygeoprocessing.raster_calculator(
            [(raw_raster, 1)], div_by_10, target_raster,
            gdal.GDT_Float32, target_nodata)


def run_WCS_exclosures(site_csv):
    """Run the model at WCS exclosure locations: NAMEM and CHIRPS precip."""
    run_dict = {
        'namem': [
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures\NAMEM",
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\WCS_exclosures\NAMEM"],
        'chirps': [
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures\CHIRPS",
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\WCS_exclosures\CHIRPS"]}

    def default_forage_args():
        args = {
            'start_month': 1,
            'start_year': 2008,
            'num_months': 132,  # Jan 2008 to Dec 2018
            'template_level': 'GH',
            'fix_file': 'drygfix.100',
            'prop_legume': 0.0,
            'steepness': 1.,
            'DOY': 1,
            'mgmt_threshold': 100.,
            'century_dir': 'C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/CENTURY4.6/Century46_PC_Jan-2014',
            'user_define_protein': 1,
            'user_define_digestibility': 0,
            'herbivore_csv': r"C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/cashmere_goats.csv",
            'grass_csv': r"C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/grass.csv",
            'restart_monthly': 0,
        }
        return args

    def modify_stocking_density(herbivore_csv, new_sd):
        """Modify the stocking density in the herbivore csv input."""
        df = pandas.read_csv(herbivore_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one herbivore type"
        df['stocking_density'] = df['stocking_density'].astype(float)
        df.set_value(0, 'stocking_density', new_sd)
        df.to_csv(herbivore_csv)

    def edit_grass_csv(grass_csv, label):
        """Edit the grass csv to reflect a new site."""
        df = pandas.read_csv(grass_csv)
        df = df.set_index(['index'])
        assert len(df) == 1, "We can only handle one grass type"
        df[['label']] = df[['label']].astype(type(label))
        df.set_value(0, 'label', label)
        df.to_csv(grass_csv)

    forage_args = default_forage_args()
    modify_stocking_density(forage_args['herbivore_csv'], 0)
    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    for precip_source in run_dict.keys():
        forage_args['input_dir'] = run_dict[precip_source][0]
        outer_outdir = run_dict[precip_source][1]
        for site in site_list:
            forage_args['latitude'] = site['latitude']
            forage_args['outdir'] = os.path.join(outer_outdir,
                                                 '{}'.format(site['site_id']))
            if not os.path.isfile(os.path.join(forage_args['outdir'],
                                               'summary_results.csv')):
                edit_grass_csv(forage_args['grass_csv'], site['site_id'])
                forage.execute(forage_args)


def summarize_biomass(site_csv, save_as):
    """Summarize simulated biomass at WCS exclosure sites."""
    run_dict = {
        'namem': [
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures\NAMEM",
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\WCS_exclosures\NAMEM"],
        'chirps': [
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures\CHIRPS",
            r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\WCS_exclosures\CHIRPS"]}
    site_list = pandas.read_csv(site_csv).to_dict(orient='records')
    df_list = []
    columns = [
        'green_biomass_gm2', 'dead_biomass_gm2', 'total_biomass_gm2',
        'year', 'month']
    for precip_source in run_dict.keys():
        outer_outdir = run_dict[precip_source][1]
        for site in site_list:
            site_id = site['site_id']
            outdir = os.path.join(outer_outdir, '{}'.format(site_id))
            sum_df = pandas.read_csv(
                os.path.join(outdir, 'summary_results.csv'))
            sum_df.set_index(['step'], inplace=True)
            subset = sum_df.loc[sum_df['month'].isin([7, 8, 9])]
            subset = subset.loc[subset['year'].isin([2016, 2017, 2018])]
            subset['green_biomass_gm2'] = (
                subset['{}_green_kgha'.format(site_id)] / 10.)
            subset['dead_biomass_gm2'] = (
                subset['{}_dead_kgha'.format(site_id)] / 10.)
            subset['total_biomass_gm2'] = (
                subset['green_biomass_gm2'] + subset['dead_biomass_gm2'])
            subset = subset[columns]
            subset['climate_source'] = precip_source
            subset['site_id'] = site_id
            df_list.append(subset)
    sum_df = pandas.concat(df_list)
    sum_df.to_csv(save_as)


if __name__ == "__main__":
    exclosure_shp = r"E:\GIS_local\Mongolia\From_Chantsa\exclosure_locations.shp"
    processing_dir = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures"
    CHIRPS_input_dir = os.path.join(processing_dir, 'CHIRPS')
    # generate_inputs_wcs_exclosures(
        # exclosure_shp, processing_dir, CHIRPS_input_dir)
    namem_precip = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\data\climate\NAMEM\NAMEM_precip.csv"
    namem_temp = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\data\climate\NAMEM\NAMEM_temp.csv"
    NAMEM_input_dir = os.path.join(processing_dir, 'NAMEM')
    # namem_to_wth(namem_precip, namem_temp, NAMEM_input_dir)
    site_stn_match_table = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\data\climate\NAMEM\exclosure_match_table.csv"
    # write_namem_sch_files(site_stn_match_table, NAMEM_input_dir)
    # copy *.100 files from CHIRPS inputs dir to NAMEM inputs dir
    site_csv = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_inputs\WCS_exclosures\soil_table.csv"
    # run_WCS_exclosures(site_csv)
    biomass_summary = r"C:\Users\ginge\Dropbox\NatCap_backup\Mongolia\model_results\WCS_exclosures\simulated_biomass_summary_2016-7-8.csv"
    summarize_biomass(site_csv, biomass_summary)
