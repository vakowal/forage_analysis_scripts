# generate inputs for the forage model
# on regional properties, Laikipia
# and from tabular weather data downloaded from NOAA
import os
import re
import collections
import random
import tempfile
import shutil

from osgeo import gdal
from osgeo import ogr

import pandas
import numpy
import pygeoprocessing


def calculate_total_annual_precip(raster_dir, zonal_shp, save_as):
    """Calculate average annual precipitation within properties identified by
    zonal_shp.  raster_dir should identify the folder of rasters, one for each
    month.  Save the table as save_as."""

    mosaicdir = os.path.join(raster_dir, "mosaic")
    if not os.path.exists(mosaicdir):
        os.makedirs(mosaicdir)

    sumdir = os.path.join(raster_dir, "sum")
    if not os.path.exists(sumdir):
        os.makedirs(sumdir)

    for m in range(1, 13):
        inputs = [os.path.join(raster_dir, "prec{}_27.tif".format(m)),
                  os.path.join(raster_dir, "prec{}_37.tif".format(m))]
        save_name = "mosaic_prec{}.tif".format(m)
        if not os.path.isfile(os.path.join(mosaicdir, save_name)):
            arcpy.MosaicToNewRaster_management(inputs, mosaicdir, save_name,
                                                   "", "32_BIT_FLOAT", "", 1,
                                                   "LAST", "")

    def sum_rasters(raster_list, save_as, cell_size):
        def sum_op(*rasters):
            return numpy.sum(numpy.array(rasters), axis=0)
        nodata = 9999
        pygeoprocessing.geoprocessing.vectorize_datasets(
                raster_list, sum_op, save_as,
                gdal.GDT_UInt16, nodata, cell_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

    raster_list = [os.path.join(mosaicdir, f) for f in os.listdir(mosaicdir)
                   if f.endswith('.tif')]
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                raster_list[0])
    sum_raster_name = os.path.join(sumdir, 'sum.tif')
    if not os.path.isfile(sum_raster_name):
        sum_rasters(raster_list, sum_raster_name, cell_size)

    # average annual precip within properties
    # calculate_zonal_averages([sum_raster_name], zonal_shp, save_as)

    # average annual precip at property centroids
    tempdir = tempfile.mkdtemp()
    point_shp = os.path.join(tempdir, 'centroid.shp')
    arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")
    rasters = [sum_raster_name]
    field_list = ['ann_precip']
    ex_list = zip(rasters, field_list)
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)
    field_list.insert(0, 'FID')
    ann_precip_dict = {'site': [], 'avg_annual_precip': []}
    with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
        for row in cursor:
            site = row[0]
            ann_precip_dict['site'].append(site)
            ann_precip_dict['avg_annual_precip'].append(row[1])
    temp_df = pandas.DataFrame.from_dict(ann_precip_dict)
    temp_df.to_csv(save_as, index=False)

def calculate_zonal_averages(raster_list, zonal_shp, save_as):
    """Calculate averages of the rasters in raster_list within zones
    identified by the zonal_shp.  Store averages in table with a row for
    each zone, identified by the file path save_as."""

    tempdir = tempfile.mkdtemp()
    zonal_raster = os.path.join(tempdir, 'zonal_raster.tif')
    field = "FID"
    # arcpy.FeatureToRaster_conversion(zonal_shp, field, zonal_raster)

    # outdir = tempfile.mkdtemp()
    # arcpy.BuildRasterAttributeTable_management(zonal_raster)
    # for raster in raster_list:
        # intermediate_table = os.path.join(outdir, os.path.basename(raster)[:-4]
                                          # + '.dbf')
        # arcpy.sa.ZonalStatisticsAsTable(zonal_raster, "Value", raster,
                                        # intermediate_table, "DATA",
                                        # statistics_type="MEAN")
    sum_dict = {}
    fornow = r"C:\Users\Ginger\Documents\ArcGIS\Default.gdb"
    arcpy.env.workspace = fornow  # outdir
    tables = arcpy.ListTables()
    for table in tables:
        sum_dict[table[:-4]] = []
        sum_dict['zone_' + table[:-4]] = []
        fields = arcpy.ListFields(table)
        field_names = ['Value', 'MEAN']
        with arcpy.da.SearchCursor(table, field_names) as cursor:
            try:
                for row in cursor:
                    sum_dict['zone_' + table[:-4]].append(row[0])
                    sum_dict[table[:-4]].append(row[1])
            except:
                import pdb; pdb.set_trace()
                print(table)
    import pdb; pdb.set_trace()
    sum_df = pandas.DataFrame.from_dict(sum_dict)
    remove_cols = [f for f in sum_df.columns.values if f.startswith('zone')]
    sum_df['zone'] = sum_df[[-1]]
    sum_df = sum_df.drop(remove_cols, axis=1)
    sum_df.to_csv(save_as, index=False)

    try:
        shutil.rmtree(tempdir)
        shutil.rmtree(outdir)
    except:
        print("Warning, temp files cannot be deleted")
        pass

def calc_soil_table(zonal_shp, save_as):
    """Make the table containing soil inputs for each property."""

    arcpy.env.overwriteOutput = 1

    soil_dir = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\Laikipia_soil_250m\averaged"
    raster_list = [os.path.join(soil_dir, f) for f in os.listdir(soil_dir)]
    calculate_zonal_averages(raster_list, zonal_shp, save_as)

def join_site_lat_long(zonal_shp, soil_table):
    """Calculate latitude and longitude of the centroid of each property and
    join it to the soil table to be used as input for site.100 file."""

    soil_df = pandas.read_csv(soil_table).set_index("zone")
    soil_df["latitude"] = "NA"
    soil_df["longitude"] = "NA"
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)  # WGS 1984
    tempdir = tempfile.mkdtemp()
    point_shp = os.path.join(tempdir, 'centroid.shp')
    arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")
    arcpy.AddXY_management(point_shp)
    with arcpy.da.SearchCursor(point_shp, ['FID', 'POINT_X', 'POINT_Y']) as \
                               cursor:
        for row in cursor:
            soil_df = soil_df.set_value([row[0]], 'longitude', row[1])
            soil_df = soil_df.set_value([row[0]], 'latitude', row[2])
    soil_df.to_csv(soil_table)

    try:
        shutil.rmtree(tempdir)
    except:
        print("Warning, temp files cannot be deleted")
        pass

def write_soil_table(site_shp, soil_dir, save_as):
    """Make soil table to use as input for site files"""

    # make a temporary copy of the point shapefile to append worldclim values
    tempdir = tempfile.mkdtemp()
    source_shp = site_shp
    site_shp = os.path.join(tempdir, 'points.shp')
    arcpy.Copy_management(source_shp, site_shp)

    rasters = [f for f in os.listdir(soil_dir) if f.endswith('.tif')]
    field_list = [r[:6] for r in rasters]
    raster_files = [os.path.join(soil_dir, f) for f in rasters]
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

    # add lat and long
    soil_df["latitude"] = "NA"
    soil_df["longitude"] = "NA"
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)  # WGS 1984
    arcpy.AddXY_management(site_shp)
    with arcpy.da.SearchCursor(site_shp, ['site_id', 'POINT_X', 'POINT_Y']) as \
                               cursor:
        for row in cursor:
            soil_df = soil_df.set_value([row[0]], 'longitude', row[1])
            soil_df = soil_df.set_value([row[0]], 'latitude', row[2])
    soil_df.to_csv(save_as)

def count_missing_values_namem(namem_precip, namem_temp):
    """How many missing values in NAMEM data for each soum center?"""

    temp_df = pandas.read_csv(namem_temp)
    prec_df = pandas.read_csv(namem_precip)
    namem_df = pandas.merge(prec_df, temp_df, how='outer')
    namem_df[['year', 'month']] = namem_df[['year', 'month']].astype(int)
    year_list = namem_df['year'].unique().tolist()
    year_list = range(min(year_list), max(year_list) + 1)
    missing_dict = {'site': [], 'missing_precip_mm': [],
                    'missing_min_temp': [], 'missing_max_temp': []}
    for site in namem_df.station_name.unique():
        missing_dict['site'].append(site)
        sub_df = namem_df.loc[(namem_df["station_name"] == site)]
        # replace missing values with average value for each month
        for year in year_list:
            if year == 2017:
                month_list = range(1, 7)
            else:
                month_list = range(1, 13)
            for month in month_list:
                test_df = sub_df.loc[(sub_df['month'] == month) &
                                     (sub_df['year'] == year)]
                if len(test_df) == 0:
                    placeholder = pandas.DataFrame({'month': [month],
                                                'year': [year]})
                    sub_df = pandas.concat([sub_df, placeholder])
        for label in ['min_temp', 'max_temp', 'precip_mm']:
            missing_count = len(sub_df.loc[sub_df[label].isnull(), label])
            missing_dict['missing_{}'.format(label)].append(missing_count)
    df = pandas.DataFrame(missing_dict)
    save_as = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\nAMEM\missing_records.csv"
    df.to_csv(save_as)

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
        sub_df = sub_df.sort(['year', 'month'])
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

def write_site_files_mongolia(
        template, soil_table, shp_id_field, save_dir):
    """Write the site.100 file to launch Century.

    Parameters:
        template (string): path to template site.100 file
        soil_table (string): path to table containing latitude, longtidue, and
            all soil values for each site
        shp_id_field (string): field name identifying sites in the soil table
        save_dir (string): path to directory where site.100 files should be
            saved

    Side effects:
        writes a site.100 table in `save_dir` for each site indicated by a row
            in `soil_table`

    Returns:
        None

    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    in_list = pandas.read_csv(soil_table).to_dict(orient="records")
    for inputs_dict in in_list:
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            first_line = (
                '{} (generated by script)\n'.format(inputs_dict[shp_id_field]))
            newfile.write(first_line)
            with open(template, 'r') as old_file:
                next(old_file)
                for line in old_file:
                    if '  \'SITLAT' in line:
                        item = '{:0.12f}'.format(inputs_dict['latitude'])[:7]
                        newline = '%s           \'SITLAT\'\n' % item
                    elif '  \'SITLNG' in line:
                        item = '{:0.12f}'.format(inputs_dict['longitude'])[:7]
                        newline = '%s           \'SITLNG\'\n' % item
                    elif '  \'SAND' in line:
                        num = inputs_dict['sndppt'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SAND\'\n' % item
                    elif '  \'SILT' in line:
                        num = inputs_dict['sltppt'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SILT\'\n' % item
                    elif '  \'CLAY' in line:
                        num = inputs_dict['clyppt'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'CLAY\'\n' % item
                    elif '  \'BULKD' in line:
                        num = inputs_dict['bldfie'] / 1000.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'BULKD\'\n' % item
                    elif '  \'PH' in line:
                        num = inputs_dict['phihox'] / 10.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'PH\'\n' % item
                    else:
                        newline = line
                    newfile.write(newline)
        save_as = os.path.join(save_dir, '{}.100'.format(inputs_dict[shp_id_field]))
        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)
        # generate weather statistics with worldclim_to_site_file() or
        # wth_to_site_file

def write_site_files(template, soil_table, save_dir):
    """Write the site.100 file for each property, using the "zone" field in the
    soil table as identifier for each property."""

    in_list = pandas.read_csv(soil_table).to_dict(orient="records")
    for inputs_dict in in_list:
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            first_line = '%s (generated by script)\n' % inputs_dict['zone']
            newfile.write(first_line)
            with open(template, 'r') as old_file:
                next(old_file)
                for line in old_file:
                    if '  \'SITLAT' in line:
                        item = '{:0.12f}'.format(inputs_dict['latitude'])[:7]
                        newline = '%s           \'SITLAT\'\n' % item
                    elif '  \'SITLNG' in line:
                        item = '{:0.12f}'.format(inputs_dict['longitude'])[:7]
                        newline = '%s           \'SITLNG\'\n' % item
                    elif '  \'SAND' in line:
                        num = inputs_dict['geonode-sndppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SAND\'\n' % item
                    elif '  \'SILT' in line:
                        num = inputs_dict['geonode-sltppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SILT\'\n' % item
                    elif '  \'CLAY' in line:
                        num = inputs_dict['geonode-clyppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'CLAY\'\n' % item
                    elif '  \'BULKD' in line:
                        item = '{:0.12f}'.format(inputs_dict[
                                       'geonode-bldfie_m_sl_0-15_div1000'])[:7]
                        newline = '%s           \'BULKD\'\n' % item
                    elif '  \'PH' in line:
                        item = '{:0.12f}'.format(inputs_dict[
                                       'geonode-phihox_m_sl_0-15_div10'])[:7]
                        newline = '%s           \'PH\'\n' % item
                    else:
                        newline = line
                    newfile.write(newline)
        save_as = os.path.join(save_dir, '{}.100'.format(
                                                     int(inputs_dict['zone'])))
        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)
        # generate weather statistics (manually :( )

def make_sch_files_mongolia(
        soil_table, shp_id_field, template_hist, template_sch, save_dir,
        site_wth_match_file=None):
    """Make schedule files for use with Century for each site in a table.

    Parameters:
        soil_table (string): path to table containing site identifier,
            latitude, longtidue, and all soil values for each site
        shp_id_field (string): field name identifying sites in the soil table
        template_hist (string): path to template schedule file for historical
            (spin-up) Century run
        template_sch (string): path to template schedule file for Century run
        save_dir (string): path to directory where schedule files should be
            written

    Side effects:
        creates a historical and current sch file in `save_dir` for each site
            indicated by a row in `soil_table`

    Returns:
        None

    """

    def copy_sch_file(template, site_name, save_as, wth_file=None):
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(template, 'r') as old_file:
                for line in old_file:
                    if '  Weather choice' in line:
                        if wth_file:
                            newfile.write('F             Weather choice\n')
                            newfile.write('{}\n'.format(wth_file))
                            line = old_file.readline()
                        else:
                            newfile.write('M             Weather choice\n')
                            line = old_file.readline()
                    if '.wth\n' in line:
                        line = old_file.readline()
                    if '  Site file name' in line:
                        item = '{:14}'.format('{}.100'.format(site_name))
                        newfile.write('{}Site file name\n'.format(item))
                    else:
                        newfile.write(line)

        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)

    site_df = pandas.read_csv(soil_table)
    site_list = site_df[shp_id_field].unique().tolist()
    if site_wth_match_file:
        match_df = pandas.read_csv(site_wth_match_file).set_index(shp_id_field)
    for site in site_list:
        save_as = os.path.join(save_dir, '{}.sch'.format(site))
        if site_wth_match_file:
            wth_stn = match_df.get_value(site, 'name_en')
            wth = '{}.wth'.format(wth_stn)
        else:
            wth = None
        copy_sch_file(template_sch, site, save_as, wth_file=wth)
        save_as = os.path.join(save_dir, '{}_hist.sch'.format(site))
        copy_sch_file(template_hist, site, save_as)

def make_sch_files(template_hist, template_extend, soil_table, save_dir):
    """Write the schedule files (hist and extend) to run each site, using the
    "zone" field in the soil table as identifier and name for each property."""

    def copy_sch_file(template, site_name, weather_file, save_as):
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(template, 'r') as old_file:
                for line in old_file:
                    if '  Site file name' in line:
                        item = '{:14}'.format('{}.100'.format(site_name))
                        newline = '{}Site file name\n'.format(item)
                    elif '.wth' in line:
                        newline = '{}\n'.format(weather_file)
                    else:
                        newline = line
                    newfile.write(newline)
        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)

    site_df = pandas.read_csv(soil_table)
    for site_name in site_df.zone:
        weather_file = '{}.wth'.format(site_name)
        save_as = os.path.join(save_dir, '{}_hist.sch'.format(site_name))
        copy_sch_file(template_hist, site_name, weather_file, save_as)
        save_as = os.path.join(save_dir, '{}.sch'.format(site_name))
        copy_sch_file(template_extend, site_name, weather_file, save_as)

def clip_rasters_arcpy(rasters_folder, clipped_folder, aoi_shp, endswith):
    """Clip large rasters to an aoi using arcpy instead of gdal."""

    to_clip = [f for f in os.listdir(rasters_folder) if f.endswith(endswith)]
    raster_nodata = 9999
    for r in to_clip:
        bil = os.path.join(rasters_folder, r)
        outExtractByMask = arcpy.sa.ExtractByMask(bil, aoi_shp)
        clipped_raster_uri = os.path.join(clipped_folder, r)
        outExtractByMask.save(clipped_raster_uri)

def clip_rasters(rasters_folder, clipped_folder, aoi_shp, endswith):
    """Clip large rasters to an aoi to speed up later processing. Rasters are
    identified as files within the rasters_folder that end with 'endswith'."""

    to_clip = [f for f in os.listdir(rasters_folder) if f.endswith(endswith)]
    raster_nodata = 9999
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                      os.path.join(rasters_folder, to_clip[0]))
    for r in to_clip:
        bil = os.path.join(rasters_folder, r)
        # arcpy.DefineProjection_management(bil, 102022)  # Africa Albers eq. area conic
        clipped_raster_uri = os.path.join(clipped_folder, r)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [bil], lambda x: x, clipped_raster_uri, gdal.GDT_Float64,
            raster_nodata, out_pixel_size, "union",
            dataset_to_align_index=0, aoi_uri=aoi_shp,
            assert_datasets_projected=False, vectorize_op=False)

def GSOM_table_to_input():
    """Convert Global Summary of the Month data tables containing precip and
    temperature to inputs for Century."""

    # GSOM_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Western_US/Kingsville_GSOM_1981_2016.csv"
    GSOM_file = r"C:\Users\Ginger\Dropbox\natCap_backup\WitW\data\Ucross\Ucross_GSOM_1980_2016.csv"
    save_as = "C:/Users/Ginger/Desktop/ucross_tripled.wth"  ## GSOM_file[:-4] + '.wth'
    gsom_df = pandas.read_csv(GSOM_file)
    gsom_df = gsom_df.sort_values(by='DATE')
    gsom_df['year'] = [int(gsom_df.iloc[r].DATE.split('-')[0]) for r in
                       range(len(gsom_df.DATE))]
    gsom_df['month'] = [int(gsom_df.iloc[r].DATE.split('-')[1]) for r in
                        range(len(gsom_df.DATE))]
    gsom_df['prec'] = gsom_df.PRCP * 3.

    # fill missing values with average values across years within months
    year_list = gsom_df['year'].unique().tolist()
    year_list = range(min(year_list), max(year_list) + 1)
    grouped = gsom_df.groupby('month')
    for year in year_list:
        for mon in range(1, 13):
            sub_df = gsom_df.loc[(gsom_df['month'] == mon) &
                                 (gsom_df['year'] == year)]
            if len(sub_df) == 0:
                placeholder = pandas.DataFrame({'month': [mon],
                                           'year': [year]})
                gsom_df = pandas.concat([gsom_df, placeholder])
    for mon in range(1, 13):
        for label in ['TMIN', 'TMAX', 'prec']:
            fill_val = grouped.get_group(mon).mean()[label]
            gsom_df.loc[(gsom_df['month'] == mon) &
                        (gsom_df[label].isnull()), label] = fill_val
    trans_dict = {'label': ['prec'] * len(year_list), 'year': year_list * 3}
    for mon in range(1, 13):
        prec = gsom_df.loc[(gsom_df["month"] == mon), "prec"].values.tolist()
        if len(prec) != len(year_list):
            import pdb; pdb.set_trace()
        trans_dict[mon] = prec
        tmin = gsom_df.loc[(gsom_df['month'] == mon), 'TMIN'].values.tolist()
        if len(tmin) < len(year_list):
            import pdb; pdb.set_trace()
        tmax = gsom_df.loc[(gsom_df['month'] == mon), 'TMAX'].values.tolist()
        if len(tmax) < len(year_list):
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
    numpy.savetxt(save_as, df.values, fmt=formats, delimiter='')

def wth_to_site_100(wth_file, site_template, save_as):
    """Calculate average weather statistics from a .wth file and write those to
    a site.100 file"""

    wth_dat = pandas.read_fwf(wth_file, header=None)
    precip = wth_dat[wth_dat[0] == 'prec']
    mean_precip = precip.iloc[:, 2:14].mean(axis=0)
    mean_precip.index = range(1, 13)
    fh, abs_path = tempfile.mkstemp()
    os.close(fh)
    with open(abs_path, 'w') as newfile:
        with open(site_template, 'r') as old_file:
            line = old_file.readline()
            while 'PRECIP(1)' not in line:
                newfile.write(line)
                line = old_file.readline()
            for m in range(1, 13):
                item = mean_precip[m]
                newline = '{:<8.5f}          \'PRECIP({})\'\n'.format(item,
                                                                      m)
                newfile.write(newline)
            for m in range(1, 13):
                newfile.write('0.00000           \'PRCSTD({})\'\n'.format(m))
            for m in range(1, 13):
                newfile.write('0.00000           \'PRCSKW({})\'\n'.format(m))
            while 'TMN2M' not in line:
                line = old_file.readline()
            while line:
                newfile.write(line)
                try:
                    line = old_file.readline()
                except StopIteration:
                    break
    shutil.copyfile(abs_path, save_as)
    os.remove(abs_path)

def clim_tables_to_inputs(prec_table, temp_table, input_folder):
    """Convert tables with monthly precipitation values (generated by the
    function process_FEWS_files) and a table with monthly temp values
    generated with process_worldclim to input files for CENTURY."""

    temp_df = pandas.read_csv(temp_table)
    prec_df = pandas.read_csv(prec_table)
    if 'year' in prec_df.columns.values:
        year_list = [2000 + y for y in prec_df.year.unique()][1:]
        year_list = [int(y) for y in year_list]
    else:
        year_list = [2015]
    for site in prec_df.site.unique():
        sub_p_df = prec_df.loc[(prec_df["site"] == site) & (
                               prec_df["month"].notnull())]
        sub_t_df = temp_df.loc[(temp_df["site"] == site)]
        sub_t_df = sub_t_df.sort_values(by='month')
        # calc average prec in month 1
        avg_mo1 = sub_p_df.loc[(sub_p_df["month"] == 1),
                               "prec"].values.mean() / 10.0
        trans_dict = {'label': ['prec'] * len(year_list),
                      'year': year_list * 3}
        for mon in range(1, 13):
            p_vals = sub_p_df.loc[(sub_p_df["month"] == mon),
                                  "prec"].values.tolist()
            trans_dict[mon] = [v / 10.0 for v in p_vals]  # RFE vals in mm
            tmin = sub_t_df.loc[(sub_t_df['month'] == mon), 'tmin'].values
            tmax = sub_t_df.loc[(sub_t_df['month'] == mon), 'tmax'].values
            trans_dict[mon].extend([tmin] * len(year_list))
            trans_dict[mon].extend([tmax] * len(year_list))
        if len(trans_dict[1]) < len(trans_dict[2]):  # RFE missing Jan-00 val
            trans_dict[1].insert(0, avg_mo1)
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
        save_as = os.path.join(input_folder, '{}.wth'.format(site))
        formats = ['%4s', '%6s'] + ['%7.2f'] * 12
        numpy.savetxt(save_as, df.values, fmt=formats, delimiter='')

def remove_wth_from_sch(input_dir):
    """To run the simulation with worldclim precipitation, must remove the
    reference to empirical weather and use just the averages in the site.100
    file."""

    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    sch_files = [os.path.join(input_dir, f) for f in sch_files]
    for sch in sch_files:
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(sch, 'r') as old_file:
                for line in old_file:
                    if '.wth' in line:
                        line = old_file.readline()
                    if "Weather choice" in line:
                        newline = "M             Weather choice\n"
                        newfile.write(newline)
                    else:
                        newfile.write(line)
        shutil.copyfile(abs_path, sch)
        os.remove(abs_path)

def remove_grazing(input_dir, out_dir):
    """Remove grazing events from schedules created to be input to the back-
    calc management routine, so that they can be supplied as inputs to run
    the simulation.  Grazing events are removed from the years 2014 and 2015,
    and the block schedule here is assumed to start with 2011."""

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    sch_files = [os.path.join(input_dir, f) for f in sch_files]
    for sch in sch_files:
        schedule_df = cent.read_block_schedule(sch)
        assert schedule_df.loc[0, 'block_start_year'] == 2011, """Schedule must
                                                     adhere to expected form"""
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(sch, 'r') as old_file:
                for line in old_file:
                    if ' GRAZ' in line:
                        if '4  ' in line or '5  ' in line:
                            line = old_file.readline()
                        else:
                            newfile.write(line)
                    else:
                        newfile.write(line)
        new_sch = os.path.join(out_dir, os.path.basename(sch))
        shutil.copyfile(abs_path, new_sch)
        os.remove(abs_path)

def process_worldclim_temp(worldclim_folder, save_as, zonal_shp=None,
                           point_shp=None):
    """Make a table of max and min monthly temperature for points which are the
    centroids of properties (features in zonal_shp, if supplied) or the points
    in point_shp, if supplied, from worldclim rasters."""

    tempdir = tempfile.mkdtemp()

    if zonal_shp and point_shp:
        raise ValueError("Only one of point or polygon layers may be supplied")

    if point_shp:
        # make a temporary copy of the point shapefile to append worldclim values
        source_shp = point_shp
        point_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, point_shp)
    if zonal_shp:
        # make property centroid shapefile to extract values to points
        point_shp = os.path.join(tempdir, 'centroid.shp')
        arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")

    # extract monthly values to each point
    rasters = [f for f in os.listdir(worldclim_folder) if f.endswith('.tif')]
    field_list = [r[10:17] for r in rasters]  # [r[:5] if r[5] == '_' else r[:6] for r in rasters]
    raster_files = [os.path.join(worldclim_folder, f) for f in rasters]
    ex_list = zip(raster_files, field_list)
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

    # read from shapefile to newly formatted table
    field_list.insert(0, 'site_id')
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # totally lazy hack
    temp_dict = {'site': [], 'month': [], 'tmin': [], 'tmax': []}
    with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
        for row in cursor:
            site = row[0]
            temp_dict['site'].extend([site] * 12)
            for f_idx in range(1, len(field_list)):
                field = field_list[f_idx]
                month = field[5:7]
                if field.startswith('tmin'):
                    temp_dict['tmin'].append(row[f_idx])
                elif field.startswith('tmax'):
                    temp_dict['tmax'].append(row[f_idx])
                else:
                    raise Exception("value not recognized")
            temp_dict['month'].extend(month_list)
    for key in temp_dict.keys():
        if len(temp_dict[key]) == 0:
            del temp_dict[key]
    temp_df = pandas.DataFrame.from_dict(temp_dict)
    temp_df.to_csv(save_as, index=False)

def process_worldclim_precip(worldclim_folder, save_as, zonal_shp=None,
                             point_shp=None):
    """Make a table of monthly precip for points which are the
    centroids of properties (features in zonal_shp, if supplied) or the points
    in point_shp, if supplied, from worldclim rasters"""

    tempdir = tempfile.mkdtemp()

    if zonal_shp and point_shp:
        raise ValueError("Only one of point or polygon layers may be supplied")

    if point_shp:
        # make a temporary copy of the point shapefile to append worldclim values
        source_shp = point_shp
        point_shp = os.path.join(tempdir, 'points.shp')
        arcpy.Copy_management(source_shp, point_shp)
    if zonal_shp:
        # make property centroid shapefile to extract values to points
        point_shp = os.path.join(tempdir, 'centroid.shp')
        arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")

    # extract monthly values to each point
    rasters = [f for f in os.listdir(worldclim_folder) if f.endswith('.tif')]
    # field_list = [r[11:12] if r[12] == '.' else r[11:13] for r in rasters]  # [r[:5] if r[5] == '_' else r[:6] for r in rasters]  # [r[10:17] for r in rasters]
    field_list = [r[15:17] for r in rasters]
    raster_files = [os.path.join(worldclim_folder, f) for f in rasters]
    ex_list = zip(raster_files, field_list)
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

    # read from shapefile to newly formatted table
    field_list.insert(0, 'site_id')
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]  # totally lazy hack
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
    prec_df.to_csv(save_as, index=False)

def process_FEWS_files(FEWS_folder, zonal_shp, prec_table):
    """Calculate precipitation at property centroids from FEWS RFE (rainfall
    estimate) rasters.  The zonal_shp shapefile should be properties."""

    # the files give dekadal (10-day) estimates, with the filename format
    # 'ea15011.bil' for the 1st period of the first month of year 2015,
    # 'ea08121.bil' for the 1st period of the 12th month of year 2008, etc

    # tempdir = tempfile.mkdtemp() todo remove
    # tempdir = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\FEWS_RFE_sum"

    # make property centroid shapefile to extract values to points
    # point_shp = os.path.join(outdir, 'centroid.shp')
    point_shp = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\regional_properties_Jul_8_2016_Mpala_split_centroid.shp"
    # arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")

    def sum_rasters(raster_list, save_as, cell_size):
        def sum_op(*rasters):
            return numpy.sum(numpy.array(rasters), axis=0)
        nodata = 9999
        pygeoprocessing.geoprocessing.vectorize_datasets(
                raster_list, sum_op, save_as,
                gdal.GDT_UInt16, nodata, cell_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

    # bil_files = [f for f in os.listdir(FEWS_folder) if f.endswith(".bil")]

    # set nodata value
    # for f in bil_files:
        # raster = os.path.join(FEWS_folder, f)
        # source_ds = gdal.Open(raster)
        # band = source_ds.GetRasterBand(1)
        # band.SetNoDataValue(9999)
        # source_ds = None
    # template = raster = os.path.join(FEWS_folder, bil_files[0])
    # cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(template)

    # calculate monthly values from dekadal (10-day) estimates
    # field_list = ['FID']
    # ex_list = []
    # for year in range(0, 16):
        # for month in range(1, 13):
            # if year == 0 and month == 1:
                # continue
            # raster_list = [os.path.join(FEWS_folder,
                                        # 'ea{:0>2}{:0>2}{}.bil'.format(
                                        # year, month, i)) for i in range(1, 4)]
            # save_as = os.path.join(tempdir, '{}_{}.tif'.format(month, year))
            # sum_rasters(raster_list, save_as, cell_size)
            # field_name = 'RFE_{:0>2}_{:0>2}'.format(month, year)
            # field_list.append(field_name)
            # ex_list.append([save_as, field_name])

    sum_folder = FEWS_folder
    monthly_rasters = [f for f in os.listdir(sum_folder) if f.endswith(".tif")]
    ex_list = [[os.path.join(sum_folder, path), path]
                for path in monthly_rasters]
    field_list = [f[:-4] for f in monthly_rasters]

    # extract monthly values to each point
    # arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

    # read monthly values into table
    num_val = len(field_list)
    field_list.insert(0, 'FID')
    prec_dict = {'site': [], 'year': [], 'month': [], 'prec': []}
    with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
        for row in cursor:
            site = row[0]
            prec_dict['site'].extend([site] * num_val)
            for f_idx in range(1, len(field_list)):
                field = field_list[f_idx]
                month = field.split('_')[0] # field[4:6]
                year = field.split('_')[1] # field[7:9]
                prec = row[f_idx]
                prec_dict['year'].append(year)
                prec_dict['month'].append(month)
                prec_dict['prec'].append(prec)
    prec_df = pandas.DataFrame.from_dict(prec_dict)
    prec_df.to_csv(prec_table)

def generate_grass_csvs(template, input_dir):
    """Make input csvs describing grass for input to the forage model. Copy a
    template, using names taken from schedule files in the input_dir."""

    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    site_list = [f[:-4] for f in sch_files]

    template_df = pandas.read_csv(template)
    for site in site_list:
        new_df = template_df.copy()
        new_df = new_df.set_value(0, 'label', site)
        save_as = os.path.join(input_dir, '{}.csv'.format(site))
        new_df.to_csv(save_as, index=False)

def generate_site_csv(input_dir, save_as):
    """Generate a csv that can be used to direct inputs to run the model."""

    def get_latitude(site_file):
        with open(site_file, 'r') as read_file:
            for line in read_file:
                if 'SITLAT' in line:
                    lat = line[:8].strip()
        return lat

    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    site_list = [f[:-4] for f in sch_files]

    site_dict = {'name': [], 'lat': []}
    for site in site_list:
        site_file = os.path.join(input_dir, '{}.100'.format(site))
        assert os.path.isfile(site_file), "file {} not found".format(site_file)
        lat = get_latitude(site_file)
        site_dict['name'].append(site)
        site_dict['lat'].append(lat)
    site_df = pandas.DataFrame(site_dict)
    site_df.to_csv(save_as, index=False)

def get_site_weather_files(schedule_file, input_dir):
    """Read filename of weather file from schedule file supplied to
    CENTURY."""

    w_file = 'NA'
    with open(schedule_file, 'r') as read_file:
        for line in read_file:
            if '.wth' in line:
                if w_file != 'NA':
                    er = "Error: two weather files found in schedule file"
                    raise Exception(er)
                else:
                    w_name = re.search('(.+?).wth', line).group(1) + '.wth'
                    w_file = os.path.join(input_dir, w_name)
    return w_file

def wth_to_site_file(soil_table, input_dir):
    """Generate weather statistics from the wth file specified in the .sch
    file, save to site file. Identify site and sch files from the soil_table
    and assume that the site.100, .sch files and .wth files are in the same
    directory, input_dir."""

    site_df = pandas.read_csv(soil_table)
    for site in site_df.site_id.unique().tolist():
        schedule_file = os.path.join(input_dir, '{}.sch'.format(site))
        wth_file = get_site_weather_files(schedule_file, input_dir)
        wth_df = pandas.read_fwf(wth_file, header=None, names=['label', 'year']
                             + range(1, 13))
        grouped = wth_df.groupby('label')
        site_file = os.path.join(input_dir, '{}.100'.format(site))
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(site_file, 'r') as old_file:
                line = old_file.readline()
                while 'PRECIP(1)' not in line:
                    newfile.write(line)
                    line = old_file.readline()
                for m in range(1, 13):
                    item = grouped.get_group('prec').mean()[m]
                    newline = '{:<8.5f}          \'PRECIP({})\'\n'.format(item,
                                                                          m)
                    newfile.write(newline)
                for m in range(1, 13):
                    newfile.write('0.00000           \'PRCSTD({})\'\n'.format(m))
                for m in range(1, 13):
                    newfile.write('0.00000           \'PRCSKW({})\'\n'.format(m))
                while 'TMN2M(1)' not in line:
                    line = old_file.readline()
                for m in range(1, 13):
                    item = grouped.get_group('tmin').mean()[m]
                    newline = '{:<8.5f}          \'TMN2M({})\'\n'.format(item,
                                                                         m)
                    newfile.write(newline)
                for m in range(1, 13):
                    item = grouped.get_group('tmax').mean()[m]
                    newline = '{:<8.5f}          \'TMX2M({})\'\n'.format(item,
                                                                         m)
                    newfile.write(newline)
                while 'TM' in line:
                    line = old_file.readline()
                while line:
                    newfile.write(line)
                    try:
                        line = old_file.readline()
                    except StopIteration:
                        break
        shutil.copyfile(abs_path, site_file)
        os.remove(abs_path)

def worldclim_to_site_file(wc_precip, wc_temp, site_file_dir):
    """Write Worldclim averages into existing site files.

    Parameters:
        wc_precip (string): path to table containing average monthly precip in
            cm for each site
        wc_temp (string): path to table containing average monthly minimum and
            maximum temperature in deg C for each site
        site_file_dir (string): path to directory containing site.100 files for
            each site. Must contain a site.100 file for each row in wc_precip
            and wc_temp

    Side effects:
        Modifies the files in site_file_dir by editing temperature and precip
            values at the top of the file

    Returns:
        None

    """
    prec_df = pandas.read_csv(wc_precip)
    temp_df = pandas.read_csv(wc_temp)
    for site in prec_df.site.unique().tolist():
        prec_subs = prec_df.loc[prec_df['site'] == site]
        prec_subs = prec_subs.set_index('month')
        temp_subs = temp_df.loc[temp_df['site'] == site]
        temp_subs = temp_subs.set_index('month')
        site_file = os.path.join(site_file_dir, '{}.100'.format(site))
        fh, abs_path = tempfile.mkstemp()
        os.close(fh)
        with open(abs_path, 'w') as newfile:
            with open(site_file, 'r') as old_file:
                line = old_file.readline()
                while 'PRECIP(1)' not in line:
                    newfile.write(line)
                    line = old_file.readline()
                for m in range(1, 13):
                    item = prec_subs.get_value(m, 'prec')
                    newline = '{:<8.5f}          \'PRECIP({})\'\n'.format(item,
                                                                          m)
                    newfile.write(newline)
                for m in range(1, 13):
                    newfile.write('0.00000           \'PRCSTD({})\'\n'.format(m))
                for m in range(1, 13):
                    newfile.write('0.00000           \'PRCSKW({})\'\n'.format(m))
                while 'TMN2M(1)' not in line:
                    line = old_file.readline()
                for m in range(1, 13):
                    item = temp_subs.get_value(m, 'tmin')
                    newline = '{:<8.5f}          \'TMN2M({})\'\n'.format(item,
                                                                         m)
                    newfile.write(newline)
                for m in range(1, 13):
                    item = temp_subs.get_value(m, 'tmax')
                    newline = '{:<8.5f}          \'TMX2M({})\'\n'.format(item,
                                                                         m)
                    newfile.write(newline)
                while 'TM' in line:
                    line = old_file.readline()
                while line:
                    newfile.write(line)
                    try:
                        line = old_file.readline()
                    except StopIteration:
                        break
        shutil.copyfile(abs_path, site_file)
        os.remove(abs_path)

def EO_to_wth(soil_table, EO_csv, year_reg, col_format, wc_temp, wc_prec,
              conversion_factor, save_dir):
    """generate .wth files from a csv of remotely sensed precipitation data,
    filling in temperature data from worldclim.  Save them in the directory
    save_dir.  The string 'col_format' describes where year and month fall
    within a column heading containing data."""

    temp_df = pandas.read_csv(wc_temp)
    prec_df = pandas.read_csv(wc_prec)
    EO_df = pandas.read_csv(EO_csv).set_index("site_id")
    EO_col = [c for c in EO_df.columns.values.tolist() if
              c.startswith(col_format[:4])]
    EO_df = EO_df[EO_col]
    year_list = []
    for col in EO_col:
        if re.search(year_reg, col):
            year_list.append(re.search(year_reg, col).group(1))
    year_list = list(set(year_list))
    site_df = pandas.read_csv(soil_table)
    site_list = site_df.site_id.unique().tolist()
    for site in site_list:
        temp_subs = temp_df.loc[temp_df['site'] == site]
        temp_subs = temp_subs.set_index('month')
        prec_subs = prec_df.loc[prec_df['site'] == site]
        prec_subs = prec_subs.set_index('month')
        trans_dict = {m: [] for m in range(1, 13)}
        trans_dict['label'] = []
        trans_dict['year'] = []
        for year in year_list:
            trans_dict['year'].append(int(year))
            trans_dict['label'].append('prec')
            for mon in range(1, 13):
                try:
                    prec = EO_df.loc[site, col_format.format(year, mon)]
                    prec = prec * conversion_factor
                except KeyError:
                    # EO data doesn't contain this month, fill with Worldclim
                    prec = prec_subs.get_value(mon, 'prec')
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


def site_file_to_precip_df(site_file):
    """Read precipitation from a site file and return a dataframe."""
    month_regex = re.compile(r'PRECIP\((.+)\)')
    precip_dict = {'month': [], 'precip': []}
    with open(site_file, 'r') as old_file:
        for line in old_file:
            if 'PRECIP(' in line:
                month = int(month_regex.search(line).group(1))
                precip = float(line[:8].strip())
                precip_dict['month'].append(month)
                precip_dict['precip'].append(precip)
    precip_df = pandas.DataFrame(precip_dict)
    return precip_df


def precip_df_to_site_file(precip_df, site_file, save_as):
    """Replace precipitation in a site file from an edited dataframe.

    Parameters:
        precip_df (dataframe): precipitation that should be added to the
            site file
        site_file (string): path to template site file that should be edited
        save_as (string): path to save new edited site file

    Returns:
        none
    """
    month_regex = re.compile(r'PRECIP\((.+)\)')
    fh, abs_path = tempfile.mkstemp()
    os.close(fh)
    with open(abs_path, 'w') as newfile:
        with open(site_file, 'r') as old_file:
            for line in old_file:
                if 'PRECIP(' in line:
                    month = int(month_regex.search(line).group(1))
                    precip = float(
                        precip_df.loc[precip_df.month == month, 'precip'])
                    line = '{:<7.4f}           \'PRECIP({})\'\n'.format(
                        precip, month)
                newfile.write(line)
    shutil.copyfile(abs_path, save_as)
    os.remove(abs_path)


def copy_non_site_files(input_dir, save_dir):
    """Copy all inputs other than site files from input_dir to save_dir."""
    files_to_move = [
        f for f in os.listdir(input_dir) if not f.endswith('.100')]
    for file in files_to_move:
        source_path = os.path.join(input_dir, file)
        destination_path = os.path.join(save_dir, file)
        shutil.copyfile(source_path, destination_path)


def measure_achieved_perturbation():
    """compare intended to achieved perturbation of rainfall."""
    def calc_pci(precip_values):
        return sum([p**2 for p in precip_values]) / (sum(precip_values)**2)

    site_csv = r"C:\Users\ginge\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\regional_properties.csv"
    input_dir = r"C:\Users\ginge\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\empty_2014_2015"
    outer_outdir = r"C:\Users\ginge\Documents\natCap\model_inputs_Kenya\regional_precip_perturbations"
    change_perc_series = [
        -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]

    sum_dict = {
        'site': [], 'tot_precip': [], 'totp_perc_change': [],
        'pci_intended_perc_change': [], 'pci': []}
    site_list = pandas.read_csv(site_csv).to_dict(orient="records")
    for mean_change_perc in change_perc_series:
        for pci_change_perc in change_perc_series:
            save_dir = os.path.join(
                outer_outdir, 'total_precip_{}_PCI_{}'.format(
                    mean_change_perc, pci_change_perc))
            for site in site_list:
                site_file = os.path.join(
                    input_dir, '{}.100'.format(int(site['name'])))
                orig_precip_df = site_file_to_precip_df(site_file)
                orig_pci = calc_pci(orig_precip_df.precip)
                edited_site_file = os.path.join(
                    save_dir, '{}.100'.format(int(site['name'])))
                edited_precip_df = site_file_to_precip_df(edited_site_file)
                edited_pci = calc_pci(edited_precip_df.precip)

                sum_dict['site'].append(site['name'])
                sum_dict['tot_precip'].append(sum(edited_precip_df.precip))
                sum_dict['totp_perc_change'].append(mean_change_perc)
                sum_dict['pci_intended_perc_change'].append(
                    pci_change_perc)
                sum_dict['pci'].append(edited_pci)
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(os.path.join(outer_outdir, 'precip_perturbations.csv'))


def laikipia_precip_experiment_workflow():
    """Generate inputs for regional properties with perturbed precip."""
    def edit_PCI(precip_df, change_perc):
        """Edit rainfall values to a series with greater or smaller PCI."""
        def calc_pci(precip_values):
            return sum([p**2 for p in precip_values]) / (sum(precip_values)**2)

        initial_pci = calc_pci(precip_df.precip)
        target_pci = initial_pci + change_perc * initial_pci
        observed_diff = target_pci - initial_pci
        edited_df = precip_df.copy()
        num_tries = 0
        while abs(observed_diff) > 0.001:
            if num_tries > 600:
                break
            edited_df['precip_rank'] = edited_df['precip'].rank(method='first')
            if observed_diff > 0:  # add precip to high precip month
                high_precip_month_rank = random.choice([11., 12.])
                low_precip_month_rank = random.choice(range(1, 11))
                change_amt = (edited_df.loc[
                    edited_df.precip_rank == low_precip_month_rank,
                    'precip'].values * 0.4)
                edited_df.loc[
                    edited_df.precip_rank == high_precip_month_rank,
                    'precip'] += change_amt
                edited_df.loc[
                    edited_df.precip_rank == low_precip_month_rank,
                    'precip'] -= change_amt
            else:  # add precip to low precip month
                high_precip_month_rank = random.choice(range(7, 13))
                low_precip_month_rank = random.choice(range(1, 7))
                change_amt = (edited_df.loc[
                    edited_df.precip_rank == high_precip_month_rank,
                    'precip'].values * 0.05)
                edited_df.loc[
                    edited_df.precip_rank == high_precip_month_rank,
                    'precip'] -= change_amt
                edited_df.loc[
                    edited_df.precip_rank == low_precip_month_rank,
                    'precip'] += change_amt
            edited_pci = calc_pci(edited_df.precip)
            observed_diff = target_pci - edited_pci
            num_tries += 1
        return edited_df

    # baseline (empirical) inputs
    site_csv = r"C:\Users\ginge\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\regional_properties.csv"
    input_dir = r"C:\Users\ginge\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\empty_2014_2015"
    outer_outdir = r"C:\Users\ginge\Documents\natCap\model_inputs_Kenya\regional_precip_perturbations"
    change_perc_series = [
        -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
    site_list = pandas.read_csv(site_csv).to_dict(orient="records")
    # edit inputs: higher, lower rainfall
    for mean_change_perc in change_perc_series:
        # increase, decrease precip concentration index
        for pci_change_perc in change_perc_series:
            save_dir = os.path.join(
                outer_outdir, 'total_precip_{}_PCI_{}'.format(
                    mean_change_perc, pci_change_perc))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for site in site_list:
                site_file = os.path.join(
                    input_dir, '{}.100'.format(int(site['name'])))
                precip_df = site_file_to_precip_df(site_file)
                precip_df.precip += precip_df.precip * mean_change_perc
                if pci_change_perc == 0:
                    edited_precip_df = precip_df.copy()
                else:
                    edited_precip_df = edit_PCI(precip_df, pci_change_perc)
                edited_site_file = os.path.join(
                    save_dir, '{}.100'.format(int(site['name'])))
                precip_df_to_site_file(
                    edited_precip_df, site_file, edited_site_file)
                copy_non_site_files(input_dir, save_dir)


def laikipia_regional_properties_workflow():
    """Nasty list of datasets and functions performed to process inputs for
    regional properties in Laikipia"""

    zonal_shp = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\regional_properties_Jul_8_2016.shp"
    soil_table = r"C:\Users\Ginger\Desktop\Soil_avg.csv"
    # calc_soil_table(zonal_shp, soil_table)
    # join_site_lat_long(zonal_shp, soil_table)
    save_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    template_100 = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\Golf_10.100"
    # write_site_files(template_100, soil_table, save_dir)
    template_hist = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0_hist.sch"
    template_extend = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0.sch"
    # make_sch_files(template_hist, template_extend, soil_table, save_dir)
    FEWS_folder = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\FEWS_RFE"
    clipped_folder = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\FEWS_RFE_clipped"
    aoi_shp = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\Laikipia_soil_250m\Laikipia_soil_clip_prj.shp"
    # clip_FEWS_files(FEWS_folder, clipped_folder, aoi_shp)
    prec_table = r"C:\Users\Ginger\Desktop\prec.csv"
    # process_FEWS_files(clipped_folder, zonal_shp, prec_table)
    worldclim_temp_folder = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\Laikipia_Worldclim_temp"
    temp_table = r"C:\Users\Ginger\Desktop\temp.csv"
    worldclim_precip_folder = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\Laikipia_Worldclim_prec\mosaic"
    # process_worldclim_precip(worldclim_precip_folder, prec_table, zonal_shp=zonal_shp)
    process_worldclim_temp(worldclim_temp_folder, temp_table, zonal_shp=zonal_shp)
    input_folder = 'C:/Users/Ginger/Desktop/test_wth'
    # clim_tables_to_inputs(prec_table, temp_table, input_folder)
    template = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\Forage_model\model_inputs\grass_suyian.csv"
    input_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    # generate_grass_csvs(template, input_dir)
    site_csv = os.path.join(input_dir, 'regional_properties.csv')
    # generate_site_csv(input_dir, site_csv)
    # remove_wth_from_sch(input_dir)
    input_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    out_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\empty_2014_2015"
    # remove_grazing(input_dir, out_dir)
    save_as = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\Data\Kenya\regional_average_temp_centroid.csv"
    # calculate_total_annual_precip(worldclim_precip_folder, zonal_shp, save_as)

def mongolia_workflow():
    """Generate climate inputs to run the model at Boogie's monitoring points
    for sustainable cashmere."""

    worldclim_tmax_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_tmax"
    worldclim_tmin_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_tmin"
    worldclim_precip_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_precip"
    clipped_outer_folder = r"C:\Users\Ginger\Documents\natCap\GIS_local\Mongolia\Worldclim"
    bounding_aoi = r"C:\Users\Ginger\Documents\natCap\GIS_local\Mongolia\From_Boogie\shapes\GK_reanalysis\monitoring_boundary.shp"

    # clip_rasters_arcpy(worldclim_tmax_folder,
                 # os.path.join(clipped_outer_folder, 'tmax'),
                 # bounding_aoi, '.tif')
    # clip_rasters_arcpy(worldclim_tmin_folder,
                 # os.path.join(clipped_outer_folder, 'tmin'),
                 # bounding_aoi, '.tif')
    # clip_rasters_arcpy(worldclim_precip_folder,
                 # os.path.join(clipped_outer_folder, 'precip'),
                 # bounding_aoi, '.tif')  # TODO both tmin and tmax should go into a folder called "temp"

    point_shp = r"E:\GIS_local\Mongolia\From_Boogie\shapes\GK_reanalysis\CBM_SCP_sites.shp"
    # point_shp = "C:/Users/Ginger/Documents/NatCap/GIS_local/Mongolia/CHIRPS/CHIRPS_pixel_centroid_2_soums.shp"
    wc_temp = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\data\climate\Worldclim\monitoring_points_temp.csv"
    # wc_temp = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\Worldclim\soum_ctrs_temp.csv"
    # process_worldclim_temp(os.path.join(clipped_outer_folder, 'temp'), wc_temp,
                           # point_shp=point_shp)
    wc_precip = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\data\climate\Worldclim\monitoring_points_precip.csv"
    # wc_precip = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\Worldclim\CHIRPS_pixels_precip.csv"
    # process_worldclim_precip(os.path.join(clipped_outer_folder, 'precip'),
                             # wc_precip, point_shp=point_shp)
    ### divide worldclim precip by 10.0 manually ###
    soil_dir = r"E:\GIS_local\Mongolia\Soilgrids_250m"
    soil_table = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\data\soil\monitoring_points_soil_isric_250m.csv"
    # soil_table = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\soil\CHIRPS_pixels_soil_isric_250m_2_soums.csv"
    # write_soil_table(point_shp, soil_dir, soil_table)
    template_100 = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\no_grazing.100"
    # site_file_dir = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\SCP_sites\chirps_prec_back_calc"
    site_file_dir = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\SCP_sites\chirps_prec_historical_schedule"
    # write_site_files_mongolia(template_100, soil_table, site_file_dir)
    # worldclim_to_site_file(wc_precip, wc_temp, site_file_dir)
    template_hist = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\historical_schedule_hist.sch"
    # template_hist = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\no_grazing_GCD_G_hist.sch"
    template_sch = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\historical_schedule.sch"
    template_sch = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\historical_schedule_no_backcalc.sch"
    # template_sch = r"C:\Users\ginge\Dropbox\natCap_backup\Mongolia\model_inputs\template_files\no_grazing_GCD_G.sch"
    make_sch_files_mongolia(soil_table, template_hist, template_sch,
                            site_file_dir)
    namem_precip = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\nAMEM\nAMEM_precip.csv"
    namem_temp = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\nAMEM\nAMEM_temp.csv"
    # site_file_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\model_inputs\soum_centers\namem_clim"
    # site_file_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\model_inputs\soum_centers\namem_clim_wc_temp"
    site_match_csv = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\nAMEM\soum_ctr_match_table.csv"
    # namem_to_wth(namem_precip, namem_temp, site_file_dir, wc_temp=wc_temp,
    #              site_match_csv=site_match_csv)
    # write_site_files_mongolia(template_100, soil_table, site_file_dir)
    # site_stn_match_table = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\summaries_GK\CBM_SCP_points_nearest_soum_ctr.csv"
    # site_stn_match_table = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\nAMEM\soum_ctr_match_table.csv"
    # make_sch_files_mongolia(soil_table, template_hist, template_sch,
                            # site_file_dir,
                            # site_wth_match_file=site_stn_match_table)
    # worldclim_to_site_file(wc_precip, wc_temp, site_file_dir)
    # wth_to_site_file(soil_table, site_file_dir)  # use this version if the it makes sense to use the .wth files for spin-up
    ## generate .wth inputs from CHIRPS
    # chirps_csv = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\CHIRPS\precip_data_per_point.csv"
    chirps_csv = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\CHIRPS\precip_data_CHIRPS_pixels.csv"
    save_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\model_inputs\CHIRPS_pixels\chirps_prec"
    year_reg = '0\.(.+?)\.'
    col_format = 'chirps-v2.0.{0}.{1:02}'
    conversion_factor = 0.1  # CHIRPS data are in mm
    # EO_to_wth(soil_table, chirps_csv, year_reg, col_format, wc_temp, wc_precip,
              # conversion_factor, save_dir)
    # copy site.100 files manually from namem_clim to chirps_prec OR
    # write_site_files_mongolia(template_100, soil_table, save_dir)
    # make_sch_files_mongolia(soil_table, template_hist, template_sch,
                            # save_dir)
    # wth_to_site_file(soil_table, save_dir)
    ## generate .wth inputs from GPM
    GPM_csv = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\data\climate\GPM\precip_data_per_point.csv"
    year_reg = 'IMERG\.([0-9]{4})0'
    col_format = '3B-MO-L.GIS.IMERG.{0}{1:02}01.V04A'
    save_dir = r"C:\Users\Ginger\Dropbox\natCap_backup\Mongolia\model_inputs\GPM_prec"
    # EO_to_wth(soil_table, GPM_csv, year_reg, col_format, wc_temp, wc_precip,
              # save_dir)
    # make a point shapefile that is every pixel of a CHIRPS raster
    raster_source = r"C:\Users\Ginger\Documents\natCap\GIS_local\Mongolia\CHIRPS\chirps-v2.0.1998.01_ex.tif"
    point_destination = r"C:\Users\Ginger\Documents\natCap\GIS_local\Mongolia\CHIRPS_pixel_centroid.shp"
    # rtp.raster_to_point_vector(raster_source, point_destination)

    # count_missing_values_namem(namem_precip, namem_temp)

def ucross_workflow():
    """generate climate inputs with doubled precip"""

    GSOM_table_to_input()
    wth_file = "C:/Users/Ginger/Desktop/ucross_tripled.wth"
    site_template = r"C:\Users\Ginger\Dropbox\natCap_backup\WitW\model_inputs\Ucross\ucross.100"
    save_as = r"C:\Users\Ginger\Dropbox\natCap_backup\WitW\model_inputs\Ucross\ucross_tripled_precip.100"
    wth_to_site_100(wth_file, site_template, save_as)

def precip_data_for_Felicia():
    """Data task for Felicia 3.13.18: annual rainfall from FEWS from property
    ceontroids, including Mpala Ranch, Research, and combined"""

    zonal_shp = r"C:\Users\Ginger\Documents\natCap\GIS_local\Kenya_forage\regional_properties_Jul_8_2016_Mpala_split.shp"
    FEWS_folder = r"E:\GIS_archive\Kenya_ticks\FEWS_RFE_sum"
    prec_table = r"C:\Users\Ginger\Dropbox\natCap_backup\Forage_model\Data\Kenya\Climate\regional_precip_2014_2015_FEWS_RFE.csv"
    process_FEWS_files(FEWS_folder, zonal_shp, prec_table)


def coordinates_at_points(point_shp_path, shp_id_field):
    """Collect latitude and longitude values from points.

    Parameters:
        point_shp_path (string): path to shapefile containing point features.
            Must be in geographic coordinates
        shp_id_field (string): field in point_shp_path identifying features

    Returns:
        a data frame with the following columns: shp_id_field, 'latitude',
            'longitude'

    """
    point_vector = ogr.Open(point_shp_path)
    point_layer = point_vector.GetLayer()

    data_dict = {
        shp_id_field: [],
        'latitude': [],
        'longitude': [],
    }
    for point_feature in point_layer:
        sample_point_geometry = point_feature.GetGeometryRef()
        data_dict[shp_id_field].append(point_feature.GetField(shp_id_field))
        data_dict['latitude'].append(sample_point_geometry.GetY())
        data_dict['longitude'].append(sample_point_geometry.GetX())
    point_layer = None
    point_vector = None

    report_table = pandas.DataFrame(data_dict)
    return report_table


def raster_values_at_points(
        point_shp_path, raster_path, band, shp_id_field, raster_field_name):
    """Collect values from a raster intersecting points in a shapefile.

    Create
    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates
        raster_path (string): path to raster containing values that should be
            extracted at points
        band (int): band index of the raster to analyze
        shp_id_field (string): field in point_shp_path identifying features
        raster_field_name (string): name to assign to the field in the data
            frame that contains values extracted from the raster

    Returns:
        a data frame with one column shp_id_field containing shp_id_field
            values of point features, and one column raster_field_name
            containing values from the raster at the point location

    """
    raster_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    point_vector = ogr.Open(point_shp_path)
    point_layer = point_vector.GetLayer()

    # build up a list of the original field names so we can copy it to report
    point_field_name_list = [shp_id_field]

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
    point_layer = None
    point_vector = None

    # each element will hold the point samples for each raster in the order of
    # `point_coord_list`
    sampled_precip_data_list = []
    for field_name in point_field_name_list:
        sampled_precip_data_list.append(
            pandas.Series(
                data=feature_attributes_fieldname_map[field_name],
                name=field_name))
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(band)
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
        pandas.Series(data=sample_list, name=raster_field_name))

    raster = None
    band = None

    # set nodata values to NA
    report_table = pandas.DataFrame(data=sampled_precip_data_list)
    report_table = report_table.transpose()
    try:
        report_table.loc[
            numpy.isclose(report_table[raster_field_name], raster_nodata),
            raster_field_name] = None
    except TypeError:
        report_table[raster_field_name] = pandas.to_numeric(
            report_table[raster_field_name], errors='coerce')
        report_table.loc[
            numpy.isclose(report_table[raster_field_name], raster_nodata),
            raster_field_name] = None
    return report_table


def Mongolia_Julian_sites_workflow():
    """Generate inputs to run Century at Julian Ahlborn's sampling sites."""
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/sampling_sites_shapefile/site_centroids.shp"
    wc_temp_table = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/worldclim_temperature.csv"
    generate_worldclim_temperature_table(
        point_shp_path,
        "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmin/wc2.0_30s_tmin_<month>.tif",
        "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmax/wc2.0_30s_tmax_<month>.tif",
        wc_temp_table)
    wc_precip_table = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/worldclim_precip.csv"
    generate_worldclim_precip_table(
        point_shp_path,
        "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_precip/wc2.0_30s_prec_<month>.tif",
        0.1, wc_precip_table)
    soil_table = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/intermediate_data/soil.csv"
    generate_soil_table(
        point_shp_path, 'site',
        "E:/GIS_local_archive/General_useful_data/soilgrids1k/BLDFIE_M_sl3_1km_ll.tif",
        "E:/GIS_local_archive/General_useful_data/soilgrids1k/CLYPPT_M_sl3_1km_ll.tif",
        "E:/GIS_local_archive/General_useful_data/soilgrids1k/SNDPPT_M_sl3_1km_ll.tif",
        "E:/GIS_local_archive/General_useful_data/soilgrids1k/SLTPPT_M_sl3_1km_ll.tif",
        "E:/GIS_local_archive/General_useful_data/soilgrids1k/PHIHOX_M_sl3_1km_ll.tif",
        soil_table)
    template_100 = "C:/Users/ginge/Dropbox/natCap_backup/Mongolia/model_inputs/template_files/historical_schedule.100"
    inputs_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/Century_inputs"
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)
    write_site_files_mongolia(template_100, soil_table, 'site', inputs_dir)
    worldclim_to_site_file(wc_precip_table, wc_temp_table, inputs_dir)
    template_hist = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/template_files/historical_schedule_hist.sch"
    template_sch = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/template_files/historical_schedule.sch"
    make_sch_files_mongolia(
        soil_table, 'site', template_hist, template_sch, inputs_dir)


def generate_soil_table(
        point_shp_path, shp_id_field, bulkd_path, clay_path, sand_path,
        silt_path, ph_path, save_as):
    """Generate a table of soil values at points.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates
        shp_id_field (string): field in point_shp_path identifying features
        bulkd_path (string): path to raster containing bulk density values
        clay_path (string): path to raster containing % clay
        sand_path (string): path to raster containing % sand
        silt_path (string): path to raster containing % silt
        ph_path (string): path to raster containing pH
        save_as (string): location where the result table should be saved

    Side effects:
        Creates a table at the location indicated by `save_as`, containing the
            following fields: 'latitude', 'longitude', 'bldfie', 'clyppt',
            'sndppt', 'sltppt', 'phihox'

    Returns:
        None

    """
    coord_df = coordinates_at_points(point_shp_path, shp_id_field)
    bulkd_df = raster_values_at_points(
        point_shp_path, bulkd_path, 1, shp_id_field, 'bldfie')
    clay_df = raster_values_at_points(
        point_shp_path, clay_path, 1, shp_id_field, 'clyppt')
    sand_df = raster_values_at_points(
        point_shp_path, sand_path, 1, shp_id_field, 'sndppt')
    silt_df = raster_values_at_points(
        point_shp_path, silt_path, 1, shp_id_field, 'sltppt')
    ph_df = raster_values_at_points(
        point_shp_path, ph_path, 1, shp_id_field, 'phihox')

    merged_df = coord_df.merge(
        bulkd_df, on=shp_id_field, suffixes=(False, False),
        validate="one_to_one")
    merged_df = merged_df.merge(
        clay_df, on=shp_id_field, suffixes=(False, False),
        validate="one_to_one")
    merged_df = merged_df.merge(
        sand_df, on=shp_id_field, suffixes=(False, False),
        validate="one_to_one")
    merged_df = merged_df.merge(
        silt_df, on=shp_id_field, suffixes=(False, False),
        validate="one_to_one")
    merged_df = merged_df.merge(
        ph_df, on=shp_id_field, suffixes=(False, False),
        validate="one_to_one")
    merged_df.to_csv(save_as, index=False)


def generate_worldclim_temperature_table(
        point_shp_path, tmin_pattern, tmax_pattern, save_as):
    """Generate a table of max and min monthly temperature from Worldclim.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates
        tmin_pattern (string): pattern that can be used to locate worlclim
            minimum temperature rasters, where '<month>' can be replaced with
            the given month. e.g.
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmin/wc2.0_30s_tmin_<month>.tif"
        tmax_pattern (string): pattern that can be used to locate worlclim
            maximum temperature rasters, where '<month>' can be replaced with
            the given month. e.g.
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmax/wc2.0_30s_tmax_<month>.tif"
        save_as (string): path to location where temperature table should be
            saved

    Side effects:
        creates a table at the location indicated by `save_as`, with four
            columns: site (site ids), month (1:12), tmin (minimum temperature),
            tmax (maximum temperature)

    Returns:
        None

    """
    tmin_df_list = []
    tmax_df_list = []
    for month in range(1, 13):
        tmin_path = tmin_pattern.replace('<month>', '%.2d' % month)
        tmin_df = raster_values_at_points(
            point_shp_path, tmin_path, 1, 'site', 'tmin')
        tmin_df['month'] = month
        tmin_df_list.append(tmin_df)
        tmax_path = tmax_pattern.replace('<month>', '%.2d' % month)
        tmax_df = raster_values_at_points(
            point_shp_path, tmax_path, 1, 'site', 'tmax')
        tmax_df['month'] = month
        tmax_df_list.append(tmax_df)
    tmin_merged_df = pandas.concat(tmin_df_list)
    tmax_merged_df = pandas.concat(tmax_df_list)
    tmin_merged_df = tmin_merged_df.merge(
        tmax_merged_df, on=['site', 'month'], suffixes=(False, False),
        validate="one_to_one")
    tmin_merged_df.to_csv(save_as, index=False)


def generate_worldclim_precip_table(
        point_shp_path, precip_pattern, multiply_factor, save_as):
    """Generate a table of monthly precipitation from Worldclim.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates
        precip_pattern (string): pattern that can be used to locate worlclim
            precipitation rasters, where '<month>' can be replaced with
            the given month. e.g.
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_precip/wc2.0_30s_prec_<month>.tif"
        save_as (string): path to location where precip table should be
            saved
        multiply_factor (float or int): factor by which values in the rasters
            should be multiplied, to get precipitation in cm

    Side effects:
        creates a table at the location indicated by `save_as`, with three
            columns: site (site ids), month (1:12), prec (precipitation in cm)

    Returns:
        None

    """
    precip_df_list = []
    for month in range(1, 13):
        precip_path = precip_pattern.replace('<month>', '%.2d' % month)
        precip_df = raster_values_at_points(
            point_shp_path, precip_path, 1, 'site', 'precip_raw')
        precip_df['prec'] = precip_df['precip_raw'] * float(multiply_factor)
        precip_df['month'] = month
        precip_df_subs = precip_df[['site', 'month', 'prec']]
        precip_df_list.append(precip_df_subs)
    precip_merged_df = pandas.concat(precip_df_list)
    precip_merged_df.to_csv(save_as, index=False)


if __name__ == "__main__":
    # laikipia_precip_experiment_workflow()
    # measure_achieved_perturbation()
    # mongolia_workflow()
    Mongolia_Julian_sites_workflow()
