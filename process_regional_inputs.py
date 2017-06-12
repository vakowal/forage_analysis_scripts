# generate inputs for the forage model
# on regional properties, Laikipia
# and from tabular weather data downloaded from NOAA

import arcpy
import pandas as pd
import numpy as np
import os
import pygeoprocessing.geoprocessing
from osgeo import gdal
import tempfile
from tempfile import mkstemp
import shutil
import sys
sys.path.append(
 r'C:\Users\Ginger\Documents\Python\rangeland_production')
import forage_century_link_utils as cent

arcpy.CheckOutExtension("Spatial")

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
            return np.sum(np.array(rasters), axis=0)
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
    temp_df = pd.DataFrame.from_dict(ann_precip_dict)
    temp_df.to_csv(save_as, index=False)
    
def calculate_zonal_averages(raster_list, zonal_shp, save_as):
    """Calculate averages of the rasters in raster_list within zones
    identified by the zonal_shp.  Store averages in table with a row for
    each zone, identified by the file path save_as."""
    
    tempdir = tempfile.mkdtemp()
    zonal_raster = os.path.join(tempdir, 'zonal_raster.tif')
    field = "FID"
    arcpy.FeatureToRaster_conversion(zonal_shp, field, zonal_raster)
    
    outdir = tempfile.mkdtemp()
    arcpy.BuildRasterAttributeTable_management(zonal_raster)
    for raster in raster_list:
        intermediate_table = os.path.join(outdir, os.path.basename(raster)[:-4]
                                          + '.dbf')
        arcpy.sa.ZonalStatisticsAsTable(zonal_raster, 'VALUE', raster,
                                        intermediate_table,
                                        statistics_type="MEAN")
    sum_dict = {}
    arcpy.env.workspace = outdir
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
                print table
    import pdb; pdb.set_trace()
    sum_df = pd.DataFrame.from_dict(sum_dict)
    remove_cols = [f for f in sum_df.columns.values if f.startswith('zone')]
    sum_df['zone'] = sum_df[[-1]]
    sum_df = sum_df.drop(remove_cols, axis=1)
    sum_df.to_csv(save_as, index=False)
    
    try:
        shutil.rmtree(tempdir)
        shutil.rmtree(outdir)
    except:
        print "Warning, temp files cannot be deleted"
        pass

def calc_soil_from_points(point_shp, save_as):
    """Make the table containing soil inputs at each point."""
    
    arcpy.env.overwriteOutput = 1
    soil_dir = point_shp  # ???
    f_list = os.listdir(soil_dir)
    ex_list = []
    for f in f_list:
        field_name = field_name #?????
        field_list.append(field_name)
        raster_name = os.path.join(soil_dir, f)
        ex_list.append([raster_name, field_name])
    
    # extract monthly values to each point
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)
    
    
def calc_soil_table(zonal_shp, save_as):
    """Make the table containing soil inputs for each property."""
    
    arcpy.env.overwriteOutput = 1
    
    soil_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_soil_250m\averaged"
    raster_list = [os.path.join(soil_dir, f) for f in os.listdir(soil_dir)]
    calculate_zonal_averages(raster_list, zonal_shp, save_as)

def join_site_lat_long(zonal_shp, soil_table):
    """Calculate latitude and longitude of the centroid of each property and
    join it to the soil table to be used as input for site.100 file."""
    
    soil_df = pd.read_csv(soil_table).set_index("zone")
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
        print "Warning, temp files cannot be deleted"
        pass
    
def write_site_files(template, soil_table, save_dir):
    """Write the site.100 file for each property, using the "zone" field in the
    soil table as identifier for each property."""
    
    in_list = pd.read_csv(soil_table).to_dict(orient="records")
    for inputs_dict in in_list:
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as newfile:
            first_line = '%s (generated by script)\r\n' % inputs_dict['zone']
            newfile.write(first_line)
            with open(template, 'rb') as old_file:
                next(old_file)
                for line in old_file:
                    if '  \'SITLAT' in line:
                        item = '{:0.12f}'.format(inputs_dict['latitude'])[:7]
                        newline = '%s           \'SITLAT\'\r\n' % item
                    elif '  \'SITLNG' in line:
                        item = '{:0.12f}'.format(inputs_dict['longitude'])[:7]
                        newline = '%s           \'SITLNG\'\r\n' % item
                    elif '  \'SAND' in line:
                        num = inputs_dict['geonode-sndppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SAND\'\r\n' % item
                    elif '  \'SILT' in line:
                        num = inputs_dict['geonode-sltppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'SILT\'\r\n' % item
                    elif '  \'CLAY' in line:
                        num = inputs_dict['geonode-clyppt_m_sl_0-15'] / 100.0
                        item = '{:0.12f}'.format(num)[:7]
                        newline = '%s           \'CLAY\'\r\n' % item
                    elif '  \'BULKD' in line:
                        item = '{:0.12f}'.format(inputs_dict[
                                       'geonode-bldfie_m_sl_0-15_div1000'])[:7]
                        newline = '%s           \'BULKD\'\r\n' % item
                    elif '  \'PH' in line:
                        item = '{:0.12f}'.format(inputs_dict[
                                       'geonode-phihox_m_sl_0-15_div10'])[:7]
                        newline = '%s           \'PH\'\r\n' % item
                    else:
                        newline = line
                    newfile.write(newline)
        save_as = os.path.join(save_dir, '{}.100'.format(
                                                     int(inputs_dict['zone'])))
        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)
        # generate weather statistics (manually :( )

def make_sch_files(template_hist, template_extend, soil_table, save_dir):
    """Write the schedule files (hist and extend) to run each site, using the
    "zone" field in the soil table as identifier and name for each property."""
    
    def copy_sch_file(template, site_name, weather_file, save_as):
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as newfile:
            with open(template, 'rb') as old_file:
                for line in old_file:
                    if '  Site file name' in line:
                        item = '{:14}'.format('{}.100'.format(site_name))
                        newline = '{}Site file name\r\n'.format(item)
                    elif '.wth' in line:
                        newline = '{}\r\n'.format(weather_file)
                    else:
                        newline = line
                    newfile.write(newline)
        shutil.copyfile(abs_path, save_as)
        os.remove(abs_path)

    site_df = pd.read_csv(soil_table)
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
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                      os.path.join(rasters_folder, to_clip[0]))
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
    GSOM_file = "C:/Users/Ginger/Dropbox/NatCap_backup/Forage_model/Data/Western_US/Ucross/Ucross_GSOM_1980_2016.csv"
    save_as = GSOM_file[:-4] + '.wth'
    gsom_df = pd.read_csv(GSOM_file)
    gsom_df = gsom_df.sort_values(by='DATE')
    gsom_df['year'] = [int(gsom_df.iloc[r].DATE.split('-')[0]) for r in
                       range(len(gsom_df.DATE))]
    gsom_df['month'] = [int(gsom_df.iloc[r].DATE.split('-')[1]) for r in
                        range(len(gsom_df.DATE))]
    gsom_df['prec'] = gsom_df.PRCP
    
    # fill missing values with average values across years within months
    year_list = gsom_df['year'].unique().tolist()
    year_list = range(min(year_list), max(year_list) + 1)
    grouped = gsom_df.groupby('month')
    for year in year_list:
        for mon in range(1, 13):
            sub_df = gsom_df.loc[(gsom_df['month'] == mon) & 
                                 (gsom_df['year'] == year)]
            if len(sub_df) == 0:
                placeholder = pd.DataFrame({'month': [mon],
                                           'year': [year]})
                gsom_df = pd.concat([gsom_df, placeholder])
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
    np.savetxt(save_as, df.values, fmt=formats, delimiter='')
        
def clim_tables_to_inputs(prec_table, temp_table, input_folder):
    """Convert tables with monthly precipitation values (generated by the 
    function process_FEWS_files) and a table with monthly temp values
    generated with process_worldclim to input files for CENTURY."""
    
    temp_df = pd.read_csv(temp_table)
    prec_df = pd.read_csv(prec_table)
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
        save_as = os.path.join(input_folder, '{}.wth'.format(site))
        formats = ['%4s', '%6s'] + ['%7.2f'] * 12
        np.savetxt(save_as, df.values, fmt=formats, delimiter='')

def remove_wth_from_sch(input_dir):
    """To run the simulation with worldclim precipitation, must remove the
    reference to empirical weather and use just the averages in the site.100
    file."""
    
    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    sch_files = [os.path.join(input_dir, f) for f in sch_files]
    for sch in sch_files:
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as newfile:
            with open(sch, 'rb') as old_file:
                for line in old_file:
                    if '.wth' in line:
                        line = old_file.next()
                    if "Weather choice" in line:
                        newline = "M             Weather choice\r\n"
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
        fh, abs_path = mkstemp()
        os.close(fh)
        with open(abs_path, 'wb') as newfile:
            with open(sch, 'rb') as old_file:
                for line in old_file:
                    if ' GRAZ' in line:
                        if '4  ' in line or '5  ' in line:
                            line = old_file.next()
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
    field_list.insert(0, 'Comment')
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
                    temp_dict['tmin'].append(row[f_idx] / 10.0)
                elif field.startswith('tmax'):
                    temp_dict['tmax'].append(row[f_idx] / 10.0)
                else:
                    raise Exception, "value not recognized"
            temp_dict['month'].extend(month_list)
    for key in temp_dict.keys():
        if len(temp_dict[key]) == 0:
            del temp_dict[key]
    temp_df = pd.DataFrame.from_dict(temp_dict)
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
    field_list = [r[10:17] for r in rasters]  # [r[:5] if r[5] == '_' else r[:6] for r in rasters]
    raster_files = [os.path.join(worldclim_folder, f) for f in rasters]
    ex_list = zip(raster_files, field_list)
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)
    
    # read from shapefile to newly formatted table
    field_list.insert(0, 'Comment')
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # totally lazy hack
    prec_dict = {'site': [], 'month': [], 'prec': []}
    with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
        for row in cursor:
            site = row[0]
            prec_dict['site'].extend([site] * 12)
            for f_idx in range(1, len(field_list)):
                field = field_list[f_idx]
                month = field[5:7]
                prec_dict['prec'].append(row[f_idx])
            prec_dict['month'].extend(month_list)
    prec_df = pd.DataFrame.from_dict(prec_dict)
    prec_df.to_csv(save_as, index=False)    

def process_FEWS_files(FEWS_folder, zonal_shp, prec_table):
    """Calculate precipitation at property centroids from FEWS RFE (rainfall
    estimate) rasters.  The zonal_shp shapefile should be properties."""
    
    # the files give dekadal (10-day) estimates, with the filename format
    # 'ea15011.bil' for the 1st period of the first month of year 2015,
    # 'ea08121.bil' for the 1st period of the 12th month of year 2008, etc
    
    # tempdir = tempfile.mkdtemp() todo remove
    tempdir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\FEWS_RFE_sum"
    
    # make property centroid shapefile to extract values to points
    point_shp = os.path.join(tempdir, 'centroid.shp')
    arcpy.FeatureToPoint_management(zonal_shp, point_shp, "CENTROID")
    
    def sum_rasters(raster_list, save_as, cell_size):
        def sum_op(*rasters):
            return np.sum(np.array(rasters), axis=0)
        nodata = 9999
        pygeoprocessing.geoprocessing.vectorize_datasets(
                raster_list, sum_op, save_as,
                gdal.GDT_UInt16, nodata, cell_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
        
    bil_files = [f for f in os.listdir(FEWS_folder) if f.endswith(".bil")]
    
    # set nodata value
    for f in bil_files:
        raster = os.path.join(FEWS_folder, f)
        source_ds = gdal.Open(raster)
        band = source_ds.GetRasterBand(1)
        band.SetNoDataValue(9999)
        source_ds = None
    template = raster = os.path.join(FEWS_folder, bil_files[0])
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(template)
    
    # calculate monthly values from dekadal (10-day) estimates
    field_list = ['FID']
    ex_list = []
    for year in range(0, 16):
        for month in range(1, 13):
            if year == 0 and month == 1:
                continue
            raster_list = [os.path.join(FEWS_folder,
                                        'ea{:0>2}{:0>2}{}.bil'.format(
                                        year, month, i)) for i in range(1, 4)]            
            save_as = os.path.join(tempdir, '{}_{}.tif'.format(month, year))
            sum_rasters(raster_list, save_as, cell_size)
            field_name = 'RFE_{:0>2}_{:0>2}'.format(month, year)
            field_list.append(field_name)
            ex_list.append([save_as, field_name])
    
    # extract monthly values to each point
    arcpy.sa.ExtractMultiValuesToPoints(point_shp, ex_list)

    # read monthly values into table
    num_val = len(field_list)
    prec_dict = {'site': [], 'year': [], 'month': [], 'prec': []}
    with arcpy.da.SearchCursor(point_shp, field_list) as cursor:
        for row in cursor:
            site = row[0]
            prec_dict['site'].extend([site] * num_val)
            for f_idx in range(len(field_list)):
                field = field_list[f_idx]
                month = field[4:6]
                year = field[7:9]
                prec = row[f_idx]
                prec_dict['year'].append(year)
                prec_dict['month'].append(month)
                prec_dict['prec'].append(prec)
    prec_df = pd.DataFrame.from_dict(prec_dict)
    prec_df.to_csv(prec_table)

def generate_grass_csvs(template, input_dir):
    """Make input csvs describing grass for input to the forage model. Copy a
    template, using names taken from schedule files in the input_dir."""
    
    sch_files = [f for f in os.listdir(input_dir) if f.endswith('.sch')]
    sch_files = [f for f in sch_files if not f.endswith('hist.sch')]
    site_list = [f[:-4] for f in sch_files]
    
    template_df = pd.read_csv(template)
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
    site_df = pd.DataFrame(site_dict)
    site_df.to_csv(save_as, index=False)

def laikipia_regional_properties_workflow():
    """Nasty list of datasets and functions performed to process inputs for
    regional properties in Laikipia"""
    
    zonal_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\regional_properties_Jul_8_2016.shp"
    # soil_table = calc_soil_table()
    soil_table = r"C:\Users\Ginger\Desktop\Soil_avg.csv"
    # join_site_lat_long(zonal_shp, soil_table)
    save_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    template_100 = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\Golf_10.100"
    # write_site_files(template_100, soil_table, save_dir)
    template_hist = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0_hist.sch"
    template_extend = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0.sch"
    # make_sch_files(template_hist, template_extend, soil_table, save_dir)
    FEWS_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\FEWS_RFE"
    clipped_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\FEWS_RFE_clipped"
    aoi_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_soil_250m\Laikipia_soil_clip_prj.shp"
    # clip_FEWS_files(FEWS_folder, clipped_folder, aoi_shp)
    prec_table = r"C:\Users\Ginger\Desktop\prec.csv"
    # process_FEWS_files(clipped_folder, zonal_shp, prec_table)
    worldclim_temp_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_Worldclim_temp"
    temp_table = r"C:\Users\Ginger\Desktop\temp.csv"
    worldclim_precip_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_Worldclim_prec"
    # process_worldclim_precip(worldclim_precip_folder, zonal_shp, prec_table)
    # process_worldclim_temp(worldclim_temp_folder, zonal_shp, temp_table)
    input_folder = 'C:/Users/Ginger/Desktop/test_wth'
    # clim_tables_to_inputs(prec_table, temp_table, input_folder)
    template = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Forage_model\model_inputs\grass_suyian.csv"
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    # generate_grass_csvs(template, input_dir)
    site_csv = os.path.join(input_dir, 'regional_properties.csv')
    # generate_site_csv(input_dir, site_csv)
    # remove_wth_from_sch(input_dir)
    input_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    out_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\empty_2014_2015"
    # remove_grazing(input_dir, out_dir)
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\regional_average_temp_centroid.csv"
    calculate_total_annual_precip(worldclim_precip_folder, zonal_shp, save_as)
    
def canete_regular_grid_workflow():
    """Generate inputs to run CENTURY on a regular grid covering the
    intervention area in the Canete basin, Peru"""
    
    points_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\climate_and_soil\worldclim_point_intervention_area.shp" # points at which to run the sim
    soil_table = calc_soil_table(points_shp, r"C:\Users\Ginger\Desktop\Soil_avg.csv")
    soil_table = r"C:\Users\Ginger\Desktop\Soil_avg.csv"
    # join_site_lat_long(zonal_shp, soil_table)
    save_dir = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip"
    template_100 = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\Golf_10.100"
    # write_site_files(template_100, soil_table, save_dir)
    template_hist = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0_hist.sch"
    template_extend = r"C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\input\regional_properties\Worldclim_precip\0.sch"
    # make_sch_files(template_hist, template_extend, soil_table, save_dir)
    FEWS_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\FEWS_RFE"
    clipped_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\FEWS_RFE_clipped"
    aoi_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_soil_250m\Laikipia_soil_clip_prj.shp"
    
def mongolia_workflow():
    """Generate climate inputs to run the model at Boogie's monitoring points
    for sustainable cashmere."""
    
    worldclim_tmax_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_tmax"
    worldclim_tmin_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_tmin"
    worldclim_precip_folder = r"E:\GIS_archive\General_useful_data\Worldclim_2.0\worldclim_precip"
    clipped_outer_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\Worldclim"
    bounding_aoi = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\Boogie_points_bounding_aoi.shp"
    
    # clip_rasters_arcpy(worldclim_tmax_folder,
                 # os.path.join(clipped_outer_folder, 'tmax'),
                 # bounding_aoi, '.tif')
    # clip_rasters_arcpy(worldclim_tmin_folder,
                 # os.path.join(clipped_outer_folder, 'tmin'),
                 # bounding_aoi, '.tif')
    # clip_rasters_arcpy(worldclim_precip_folder,
                 # os.path.join(clipped_outer_folder, 'precip'),
                 # bounding_aoi, '.tif')
    
    point_shp = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\From_Boogie\shapes\monitoring_points.shp"
    save_as = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\Worldclim\monitoring_points_temp.csv"
    # process_worldclim_temp(os.path.join(clipped_outer_folder, 'temp'), save_as,
                           # point_shp=point_shp)
    
    save_as = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\Worldclim\monitoring_points_precip.csv"
    process_worldclim_precip(os.path.join(clipped_outer_folder, 'precip'),
                             save_as, point_shp=point_shp)
                             
if __name__ == "__main__":
    GSOM_table_to_input()
    