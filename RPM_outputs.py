"""Post process outputs from RPM."""
import os
import statistics
import shutil
import tempfile
import collections

import numpy
import pandas

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing



def raster_values_at_points(
        point_shp_path, shp_id_field, raster_path, band, raster_field_name):
    """Collect values from a raster intersecting points in a shapefile.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates and must have a field identifying sites, shp_id_field
        shp_id_field (string): field in point_shp_path identifying features
        raster_path (string): path to raster containing values that should be
            extracted at points
        band (int): band index of the raster to analyze
        raster_field_name (string): name to assign to the field in the data
            frame that contains values extracted from the raster

    Returns:
        a pandas data frame with one column 'shp_id_field' containing values
            of the `shp_id_field` from point features, and one column
            'raster_field_name' containing values from the raster at the point
            location

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
        try:
            sample_list.append(
                band.ReadAsArray(raster_x, raster_y, 1, 1)[0, 0])
        except TypeError:  # access window out of range
            sample_list.append('NA')
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
        try:
            report_table[raster_field_name] = pandas.to_numeric(
                report_table[raster_field_name], errors='coerce')
            report_table.loc[
                numpy.isclose(report_table[raster_field_name], raster_nodata),
                raster_field_name] = None
        except TypeError:
            report_table.loc[
                pandas.isnull(report_table[raster_field_name]),
                raster_field_name] = None
    return report_table


def summarize_pixel_distribution(raster_path):
    """Summarize the distribution of pixel values in a raster.

    Convert all valid pixel values to a vector of values and summarize the
    distribution of values. Calculate the median, standard deviation, and range
    of values.

    Args:
        raster_path (string): path to raster that should be summarized

    Returns:
        dictionary with keys: 'mean', 'median', 'stdev', and 'range'

    """
    value_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    value_raster = gdal.OpenEx(raster_path)
    value_band = value_raster.GetRasterBand(1)

    try:
        value_list = []
        last_blocksize = None
        for block_offset in pygeoprocessing.iterblocks(
                (raster_path, 1), offset_only=True):
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

            if last_blocksize != blocksize:
                value_array = numpy.zeros(
                    blocksize,
                    dtype=pygeoprocessing._gdal_to_numpy_type(value_band))
                last_blocksize = blocksize

            value_data = block_offset.copy()
            value_data['buf_obj'] = value_array
            value_band.ReadAsArray(**value_data)

            valid_mask = (~numpy.isclose(value_array, value_nodata))
            value_list = (
                value_list + (value_array[valid_mask].flatten().tolist()))
    finally:
        value_band = None
        gdal.Dataset.__swig_destroy__(value_raster)

    if len(value_list) > 0:
        summary_dict = {
            'mean': statistics.mean(value_list),
            'median': statistics.median(value_list),
            'stdev': statistics.stdev(value_list),
            'min': min(value_list),
            'max': max(value_list),
        }
    else:
        summary_dict = {
            'mean': 'NA',
            'median': 'NA',
            'stdev': 'NA',
            'min': 'NA',
            'max': 'NA',
        }
    return summary_dict


def average_value_in_aoi(raster_list, aoi_path):
    """Calculate the average value in a list of rasters inside an aoi.

    Args:
        raster_list (list): list of paths to rasters that should be summarized
        aoi_path (string): path to polygon vector defining the aoi. Should have
            a single feature

    Returns:
        average value across pixels within the aoi across rasters in
            raster_list
    """
    running_sum = 0
    running_count = 0
    for path in raster_list:
        zonal_stat_dict = pygeoprocessing.zonal_statistics((path, 1), aoi_path)
        if len([*zonal_stat_dict]) > 1:
            raise ValueError("Vector path contains >1 feature")
        running_sum = running_sum + zonal_stat_dict[0]['sum']
        running_count = running_count + zonal_stat_dict[0]['count']
    try:
        mean_value = float(running_sum) / running_count
    except ZeroDivisionError:
        mean_value = 'NA'
    return mean_value


def num_nodata_pixels(raster_list, aoi_path):
    """Count number of nodata values across a list of rasters inside an aoi.

    Args:
        raster_list (list): list of paths to rasters that should be summarized
        aoi_path (string): path to polygon vector defining the aoi. Should have
            a single feature

    Returns:
        number of pixels with valid values within the aoi across rasters in
            raster_list
    """
    running_count = 0
    for path in raster_list:
        zonal_stat_dict = pygeoprocessing.zonal_statistics((path, 1), aoi_path)
        if len([*zonal_stat_dict]) > 1:
            raise ValueError("Vector path contains >1 feature")
        running_count = running_count + zonal_stat_dict[0]['nodata_count']
    return running_count


def raster_list_sum(
        raster_list, input_nodata, target_path, target_nodata,
        nodata_remove=False):
    """Calculate the sum per pixel across rasters in a list.

    Sum the rasters in `raster_list` element-wise, allowing nodata values
    in the rasters to propagate to the result or treating nodata as zero. If
    nodata is treated as zero, areas where all inputs are nodata will be nodata
    in the output.

    Args:
        raster_list (list): list of paths to rasters to sum
        input_nodata (float or int): nodata value in the input rasters
        target_path (string): path to location to store the result
        target_nodata (float or int): nodata value for the result raster
        nodata_remove (bool): if true, treat nodata values in input
            rasters as zero. If false, the sum in a pixel where any input
            raster is nodata is nodata.

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_sum_op(*raster_list):
        """Add the rasters in raster_list without removing nodata values."""
        invalid_mask = numpy.any(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        sum_of_rasters[invalid_mask] = target_nodata
        return sum_of_rasters

    def raster_sum_op_nodata_remove(*raster_list):
        """Add the rasters in raster_list, treating nodata as zero."""
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        sum_of_rasters[invalid_mask] = target_nodata
        return sum_of_rasters

    if nodata_remove:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in raster_list], raster_sum_op_nodata_remove,
            target_path, gdal.GDT_Float32, target_nodata)

    else:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in raster_list], raster_sum_op,
            target_path, gdal.GDT_Float32, target_nodata)


def min_value_per_pixel(raster_list, target_path):
    """Identify, for each pixel, the minimum value across a list of rasters.

    The minimum value is calculated only for pixels with valid data across each
    raster in `raster_list`.  All rasters in `raster_list` must share nodata
    value.

    Args:
        raster_list (list): a list of paths to rasters containing the same
            quantity over the same area
        target_path (string): path where the result should be stored

    Side effects:
        creates a raster at `target_path` containing the minimum value for each
            pixel across the rasters in `raster_list`

    """
    def min_op(*raster_list):
        """Find the minimum value per pixel across rasters."""
        invalid_mask = numpy.any(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        min_val = numpy.amin(raster_list, axis=0)
        min_val[invalid_mask] = input_nodata
        return min_val

    input_nodata = pygeoprocessing.get_raster_info(raster_list[0])['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list], min_op, target_path,
        gdal.GDT_Float32, input_nodata)


def max_value_per_pixel(raster_list, target_path):
    """Identify, for each pixel, the maximum value across a list of rasters.

    The maximum value is calculated only for pixels with valid data across each
    raster in `raster_list`.  All rasters in `raster_list` must share nodata
    value.

    Args:
        raster_list (list): a list of paths to rasters containing the same
            quantity over the same area
        target_path (string): path where the result should be stored

    Side effects:
        creates a raster at `target_path` containing the maximum value for each
            pixel across the rasters in `raster_list`

    """
    def max_op(*raster_list):
        """Find the maximum value per pixel across rasters."""
        invalid_mask = numpy.any(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        max_val = numpy.amax(raster_list, axis=0)
        max_val[invalid_mask] = input_nodata
        return max_val

    input_nodata = pygeoprocessing.get_raster_info(raster_list[0])['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list], max_op, target_path,
        gdal.GDT_Float32, input_nodata)


def delete_intermediate_rasters(base_data):
    """Delete rasters created during summary."""
    for run_id in base_data['run_list']:
        for output_bn in base_data['output_list']:
            for year in base_data['year_list']:
                yearly_mean_path = os.path.join(
                    base_data['summary_output_dir'],
                    'yearly_mean_{}_{}_{}.tif'.format(output_bn, year, run_id))
                os.remove(yearly_mean_path)
                perc_change_path = os.path.join(
                    base_data['summary_output_dir'],
                    'perc_change_yearly_mean_{}_{}_{}.tif'.format(
                        output_bn, year, run_id))
                if run_id != 'A':
                    os.remove(perc_change_path)


def summarize_outputs(base_data):
    """Summarize outputs from runs of RPM."""
    def perc_change(baseline_ar, scenario_ar):
        """Calculate percent change from baseline."""
        valid_mask = (
            (~numpy.isclose(baseline_ar, input_nodata)) &
            (~numpy.isclose(scenario_ar, input_nodata)))
        result = numpy.empty(baseline_ar.shape, dtype=numpy.float32)
        result[:] = input_nodata
        result[valid_mask] = (
            (scenario_ar[valid_mask] - baseline_ar[valid_mask]) /
            baseline_ar[valid_mask] * 100)
        return result

    mean_val_dict = {
        'run_id': [],
        'output': [],
        'year': [],
        'pixel_mean': [],
    }

    diet_sufficiency_summary_dict = {
        'run_id': [],
        'year': [],
        'month': [],
        'aggregation_method': [],
        'pixel_mean': [],
    }

    perc_change_summary_dict = {
        'run_id': [],
        'output': [],
        'year': [],
        'mean_perc_change': [],
        'min_perc_change': [],
        'max_perc_change': [],
    }
    for run_id in base_data['run_list']:
        run_output_dir = os.path.join(
            base_data['outer_dir'], run_id, 'output')
        for output_bn in base_data['output_list']:
            for year in base_data['year_list']:
                year_raster_list = [
                    os.path.join(run_output_dir, '{}_{}_{}.tif').format(
                        output_bn, year, month) for month in range(1, 13)]
                input_nodata = pygeoprocessing.get_raster_info(
                    year_raster_list[0])['nodata'][0]
                yearly_mean_path = os.path.join(
                    base_data['summary_output_dir'],
                    'yearly_mean_{}_{}_{}.tif'.format(output_bn, year, run_id))
                raster_list_mean(
                    year_raster_list, input_nodata, yearly_mean_path,
                    input_nodata)

                # descriptive statistics: monthly average across pixels
                stat_df = summarize_pixel_distribution(yearly_mean_path)
                mean_val_dict['run_id'].append(run_id)
                mean_val_dict['output'].append(output_bn)
                mean_val_dict['year'].append(year)
                mean_val_dict['pixel_mean'].append(stat_df['mean'])

        # number of months where average diet sufficiency across aoi was > 1
        for year in base_data['year_list']:
            for month in range(1, 13):
                output_path = os.path.join(
                    run_output_dir, 'diet_sufficiency_{}_{}.tif').format(
                        year, month)
                zonal_stat_dict = pygeoprocessing.zonal_statistics(
                    (output_path, 1), base_data['aoi_path'])
                try:
                    mean_value = (
                        float(zonal_stat_dict[0]['sum']) /
                        zonal_stat_dict[0]['count'])
                except ZeroDivisionError:
                    mean_value = 'NA'
                diet_sufficiency_summary_dict['run_id'].append(run_id)
                diet_sufficiency_summary_dict['year'].append(year)
                diet_sufficiency_summary_dict['month'].append(month)
                diet_sufficiency_summary_dict['aggregation_method'].append(
                    'average_across_pixels')
                diet_sufficiency_summary_dict['pixel_mean'].append(mean_value)

    # summarize percent change from baseline
    for run_id in base_data['run_list']:
        if run_id == 'A':
            continue
        run_output_dir = os.path.join(
            base_data['outer_dir'], run_id, 'output')
        for output_bn in base_data['output_list']:
            for year in base_data['year_list']:
                baseline_path = os.path.join(
                    base_data['summary_output_dir'],
                    'yearly_mean_{}_{}_A.tif'.format(output_bn, year))
                scenario_path = os.path.join(
                    base_data['summary_output_dir'],
                    'yearly_mean_{}_{}_{}.tif'.format(output_bn, year, run_id))
                perc_change_path = os.path.join(
                    base_data['summary_output_dir'],
                    'perc_change_yearly_mean_{}_{}_{}.tif'.format(
                        output_bn, year, run_id))
                pygeoprocessing.raster_calculator(
                    [(path, 1) for path in [baseline_path, scenario_path]],
                    perc_change, perc_change_path, gdal.GDT_Float32,
                    input_nodata)
                # descriptive statistics: monthly average across pixels
                stat_df = summarize_pixel_distribution(perc_change_path)
                perc_change_summary_dict['run_id'].append(run_id)
                perc_change_summary_dict['output'].append(output_bn)
                perc_change_summary_dict['year'].append(year)
                perc_change_summary_dict['mean_perc_change'].append(
                    stat_df['mean'])
                perc_change_summary_dict['min_perc_change'].append(
                    stat_df['min'])
                perc_change_summary_dict['max_perc_change'].append(
                    stat_df['max'])

    summary_df = pandas.DataFrame(mean_val_dict)
    save_as = os.path.join(
        base_data['summary_output_dir'], 'average_value_summary.csv')
    summary_df.to_csv(save_as, index=False)

    diet_suff_df = pandas.DataFrame(diet_sufficiency_summary_dict)
    save_as = os.path.join(
        base_data['summary_output_dir'], 'monthly_diet_suff_summary.csv')
    diet_suff_df.to_csv(save_as, index=False)

    perc_change_df = pandas.DataFrame(perc_change_summary_dict)
    save_as = os.path.join(
        base_data['summary_output_dir'], 'perc_change_summary.csv')
    perc_change_df.to_csv(save_as, index=False)


def collect_monthly_values():
    """Collect debugging outputs monthly."""
    def monthly_op(base_data):
        summary_dict = {
            'run_id': [],
            'year': [],
            'month': [],
            'output': [],
            'mean_val': [],
        }
        for run_id in base_data['run_list']:
            run_output_dir = os.path.join(
                base_data['outer_dir'], run_id, 'output')
            for output_bn in ['standing_biomass', 'diet_sufficiency']:
                for year in base_data['year_list']:
                    for month in range(1, 13):
                        raster_path = os.path.join(
                            run_output_dir, '{}_{}_{}.tif').format(
                                output_bn, year, month)
                        try:
                            zstat_dict = pygeoprocessing.zonal_statistics(
                                (raster_path, 1), base_data['aoi_path'])
                        except ValueError:
                            continue
                        try:
                            mean_val = (
                                float(zstat_dict[0]['sum']) /
                                zstat_dict[0]['count'])
                        except ZeroDivisionError:
                            mean_val = 'NA'
                        summary_dict['run_id'].append(run_id)
                        summary_dict['year'].append(year)
                        summary_dict['month'].append(month)
                        summary_dict['output'].append(output_bn)
                        summary_dict['mean_val'].append(mean_val)
            # for debug_bn in ['intake', 'emaint']:
            #     for year in base_data['year_list']:
            #         for month in range(1, 13):
            #             raster_path = os.path.join(
            #                 run_output_dir, '{}_{}.tif').format(
            #                     debug_bn, month)
            #             try:
            #                 zstat_dict = pygeoprocessing.zonal_statistics(
            #                     (raster_path, 1), base_data['aoi_path'])
            #             except ValueError:
            #                 continue
            #             try:
            #                 mean_val = (
            #                     float(zstat_dict[0]['sum']) /
            #                     zstat_dict[0]['count'])
            #             except ZeroDivisionError:
            #                 mean_val = 'NA'
            #             summary_dict['run_id'].append(run_id)
            #             summary_dict['year'].append(year)
            #             summary_dict['month'].append(month)
            #             summary_dict['output'].append(debug_bn)
            #             summary_dict['mean_val'].append(mean_val)
            summary_df = pandas.DataFrame(summary_dict)
            save_as = os.path.join(
                base_data['summary_output_dir'],
                'monthly_value_summary.csv')
            summary_df.to_csv(save_as, index=False)

    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_scenarios/Nov62020"
    base_data = {
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 12,
        'run_list': ['A', 'B', 'D', 'F'],  #, 'C', 'D', 'F', 'I'],
        'year_list': [2016]  # , 2017],  # years to calculate yearly averages for
    }
    summary_outer_dir = "C:/Users/ginge/Desktop/debugging"
    aoi_pattern = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_centroid_buffer_sfu/aoi_<idx>.shp"
    for aoi_idx in [1, 4, 11, 15]:  # range(1, 16):
        base_data['aoi_path'] = aoi_pattern.replace('<idx>', str(aoi_idx))
        base_data['outer_dir'] = os.path.join(
            outer_dir, 'aoi_{}'.format(aoi_idx))
        base_data['summary_output_dir'] = os.path.join(
            summary_outer_dir, 'aoi_{}'.format(aoi_idx))
        if not os.path.exists(base_data['summary_output_dir']):
            os.makedirs(base_data['summary_output_dir'])
        monthly_op(base_data)


def summarize_Gobi_scenarios():
    # RPM args pertaining to Gobi scenarios and what to summarize
    base_data = {
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'aoi_path': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/soums_monitoring_area_diss.shp",
        'outer_dir': "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/RPM_scenarios/revised_9.8.20",
        'summary_output_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/RPM_scenarios/revised_9.8.20",
        'run_list': ['A', 'B', 'C', 'D', 'F', 'G', 'J'],
        'year_list': [2016, 2017],  # years to calculate yearly averages for
        'output_list': ['standing_biomass', 'diet_sufficiency'],  # outputs to summarize
    }
    summarize_outputs(base_data)


def summarize_Ahlborn_scenarios():
    # RPM args pertaining to scenario runs at Julian Ahlborn's sites
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_scenarios/Dec102020"
    base_data = {
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'run_list': [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
        'year_list': [2017],  # years to calculate yearly averages for
        'output_list': ['standing_biomass', 'diet_sufficiency'],  # outputs to summarize
    }
    summary_outer_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/Ahlborn_scenarios"
    aoi_pattern = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_centroid_buffer_sfu/aoi_<idx>.shp"
    for aoi_idx in range(1, 16):
        base_data['aoi_path'] = aoi_pattern.replace('<idx>', str(aoi_idx))
        base_data['outer_dir'] = os.path.join(
            outer_dir, 'aoi_{}'.format(aoi_idx))
        base_data['summary_output_dir'] = os.path.join(
            summary_outer_dir, 'aoi_{}'.format(aoi_idx))
        summarize_outputs(base_data)
        delete_intermediate_rasters(base_data)


def summarize_Kenya_scenarios():
    """RPM scenarios in Laikipia Kenya."""
    base_data = {
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'aoi_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/Laikipia.shp",
        'outer_dir': "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_scenarios",
        'summary_output_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_results/Laikipia_RPM/RPM_scenarios",
        'run_list': ['A', 'B', 'C', 'D', 'F', 'G', 'I', 'J', 'L'],
        'year_list': [2016, 2017],  # years to calculate yearly averages for
        'output_list': ['standing_biomass', 'diet_sufficiency'],  # outputs to summarize
    }
    summarize_outputs(base_data)
    delete_intermediate_rasters(base_data)


def raster_list_mean(
        raster_list, input_nodata, target_path, target_nodata,
        nodata_remove=False):
    """Calculate the mean per pixel across rasters in a list.

    Sum the rasters in `raster_list` element-wise, allowing nodata values
    in the rasters to propagate to the result or treating nodata as zero. If
    nodata is treated as zero, areas where all inputs are nodata will be nodata
    in the output.

    Parameters:
        raster_list (list): list of paths to rasters to sum
        input_nodata (float or int): nodata value in the input rasters
        target_path (string): path to location to store the result
        target_nodata (float or int): nodata value for the result raster
        nodata_remove (bool): if true, treat nodata values in input
            rasters as zero. If false, the sum in a pixel where any input
            raster is nodata is nodata.

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_mean_op(*raster_list):
        """Add the rasters in raster_list without removing nodata values."""
        invalid_mask = numpy.any(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        mean_of_rasters = sum_of_rasters / len(raster_list)
        mean_of_rasters[invalid_mask] = target_nodata
        return mean_of_rasters

    def raster_mean_op_nodata_remove(*raster_list):
        """Add the rasters in raster_list, treating nodata as zero."""
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        mean_of_rasters = sum_of_rasters / len(raster_list)
        mean_of_rasters[invalid_mask] = target_nodata
        return mean_of_rasters

    if nodata_remove:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in raster_list], raster_mean_op_nodata_remove,
            target_path, gdal.GDT_Float32, target_nodata)

    else:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in raster_list], raster_mean_op,
            target_path, gdal.GDT_Float32, target_nodata)


def raster_difference(
        raster1, raster1_nodata, raster2, raster2_nodata, target_path,
        target_nodata, nodata_remove=False):
    """Subtract raster2 from raster1.

    Subtract raster2 from raster1 element-wise, allowing nodata values in the
    rasters to propagate to the result or treating nodata as zero.

    Parameters:
        raster1 (string): path to raster from which to subtract raster2
        raster1_nodata (float or int): nodata value in raster1
        raster2 (string): path to raster which should be subtracted from
            raster1
        raster2_nodata (float or int): nodata value in raster2
        target_path (string): path to location to store the difference
        target_nodata (float or int): nodata value for the result raster
        nodata_remove (bool): if true, treat nodata values in input
            rasters as zero. If false, the difference in a pixel where any
            input raster is nodata is nodata.

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_difference_op(raster1, raster2):
        """Subtract raster2 from raster1 without removing nodata values."""
        valid_mask = (
            (~numpy.isclose(raster1, raster1_nodata)) &
            (~numpy.isclose(raster2, raster2_nodata)))
        result = numpy.empty(raster1.shape, dtype=numpy.float32)
        result[:] = target_nodata
        result[valid_mask] = raster1[valid_mask] - raster2[valid_mask]
        return result

    def raster_difference_op_nodata_remove(raster1, raster2):
        """Subtract raster2 from raster1, treating nodata as zero."""
        numpy.place(raster1, numpy.isclose(raster1, raster1_nodata), [0])
        numpy.place(raster2, numpy.isclose(raster2, raster2_nodata), [0])
        result = raster1 - raster2
        return result

    if nodata_remove:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in [raster1, raster2]],
            raster_difference_op_nodata_remove, target_path, gdal.GDT_Float32,
            target_nodata)
    else:
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in [raster1, raster2]],
            raster_difference_op, target_path, gdal.GDT_Float32,
            target_nodata)


def summarize_eastern_steppe_scenarios():
    """Calculate difference in mean standing biomass and diet sufficiency.

    For one future climate scenario for the eastern steppe, calculate mean
    standing biomass and mean diet sufficiency over months of the year. Do the
    same for current conditions and calculate the difference in each, per
    pixel.

    """
    current_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/current/RPM_workspace"
    future_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/CanESM5_ssp370_2061-2080/RPM_workspace"
    output_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/eastern_steppe_regular_grid/RPM_results"

    year = 2017  # summarize outputs for the second year of scenario conditions
    temp_dir = tempfile.mkdtemp()
    for output in ['standing_biomass', 'diet_sufficiency']:
        current_raster_list = [
            os.path.join(current_dir, 'output', '{}_{}_{}.tif'.format(
                output, year, m)) for m in range(1, 13)]
        input_nodata = pygeoprocessing.get_raster_info(
            current_raster_list[0])['nodata'][0]
        current_mean_path = os.path.join(
            temp_dir, 'current_mean_{}.tif'.format(output))
        raster_list_mean(
            current_raster_list, input_nodata, current_mean_path, input_nodata)

        future_raster_list = [
            os.path.join(future_dir, 'output', '{}_{}_{}.tif'.format(
                output, year, m)) for m in range(1, 13)]
        future_mean_path = os.path.join(
            temp_dir, 'future_mean_{}.tif'.format(output))
        raster_list_mean(
            future_raster_list, input_nodata, future_mean_path, input_nodata)

        diff_path = os.path.join(
            output_dir,
            'mean_{}_future_minus_current_CanESM5_ssp370_2061-2080.tif'.format(
                output))
        raster_difference(
            future_mean_path, input_nodata, current_mean_path, input_nodata,
            diff_path, input_nodata)

    shutil.rmtree(temp_dir)


def eastern_steppe_time_series():
    """Make biomass time series from eastern steppe results."""
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/time_series_points.shp"
    output_dir = "C:/Users/ginge/Desktop/debugging/future_24months/RPM_workspace/output"
    df_list = []
    for year in [2016, 2017]:
        for m in range(1, 13):
            output_path = os.path.join(
                output_dir, 'standing_biomass_{}_{}.tif'.format(year, m))
            month_df = raster_values_at_points(
                point_shp_path, 'id', output_path, 1, 'standing_biomass')
            month_df['year'] = year
            month_df['month'] = m
            df_list.append(month_df)
    sum_df = pandas.concat(df_list)
    save_as = "C:/Users/ginge/Desktop/debugging/future_24months/time_series_points_standing_biomass.csv"
    sum_df.to_csv(save_as, index=False)


def ahlborn_ndvi_time_series():
    """Make time series of NDVI at Julian's sites."""
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/sampling_sites_shapefile/site_centroids.shp"
    ndvi_pattern = "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_Ahlborn_sites/exported_to_tif/ndvi_{}_{:02}.tif"
    df_list = []
    for year in [2014, 2015]:
        for m in range(1, 13):
            ndvi_path = ndvi_pattern.format(year, m)
            points_df = raster_values_at_points(
                point_shp_path, 'site', ndvi_path, 1, 'ndvi')
            points_df['year'] = year
            points_df['month'] = m
            df_list.append(points_df)
    ndvi_df = pandas.concat(df_list)
    save_as = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/biomass_vs_NDVI/ndvi_time_series_ahlborn_sites.csv"
    ndvi_df.to_csv(save_as, index=False)


def main():
    # summarize_Gobi_scenarios()
    # summarize_Ahlborn_sites()
    # summarize_Ahlborn_scenarios()
    # summarize_Kenya_scenarios()
    # collect_monthly_values()
    # summarize_eastern_steppe_scenarios()
    # eastern_steppe_time_series()
    ahlborn_ndvi_time_series()


if __name__ == "__main__":
    main()
