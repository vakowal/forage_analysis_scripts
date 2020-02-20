"""Run RPM with initial values from sampling points."""
import os
import collections
import shutil
import tempfile

import numpy
import pandas

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing
from rangeland_production import forage


# shapefiles at which we want RPM outputs summarized
SAMPLE_PATH_LIST = [
    "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/summaries_GK/master_veg_coords_2017.shp",
    "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/summaries_GK/master_veg_coords_2018.shp",
    "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/summaries_GK/master_veg_coords_2019.shp",
]

# input
DATA_DIR = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century"

def generate_base_args(workspace_dir):
    """Generate an args dict of basic inputs for rangeland production model."""
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 9,
        'starting_year': 2016,
        'n_months': 39,  # 16 for comparison with NDVI, 39 to run all the way through 2019
        'aoi_path': os.path.join(
            DATA_DIR, 'soums_monitoring_area_diss.shp'),
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            DATA_DIR, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            DATA_DIR, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            DATA_DIR, 'soil', 'phihox_sl3.tif'),
        'clay_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'sand.tif'),
        'monthly_precip_path_pattern': os.path.join(
            DATA_DIR, 'CHIRPS_div_by_10',
            'chirps-v2.0.<year>.<month>.tif'),
        'min_temp_path_pattern': os.path.join(
            DATA_DIR, 'temp', 'wc2.0_30s_tmin_<month>.tif'),
        'max_temp_path_pattern': os.path.join(
            DATA_DIR, 'temp', 'wcs2.0_30s_tmax_<month>.tif'),
        'monthly_vi_path_pattern': os.path.join(
            DATA_DIR, 'NDVI', 'ndvi_<year>_<month>.tif'),
        'site_param_table': os.path.join(
            DATA_DIR, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            DATA_DIR, 'site_index_masked_tes_gobi_!=13_1.tif'),
        'veg_trait_path': os.path.join(DATA_DIR, 'pft_trait.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            DATA_DIR, 'pft_masked_tes_gobi_!=13_<PFT>.tif'),
        'animal_trait_path': os.path.join(
            DATA_DIR, 'animal_trait_table.csv'),
        'animal_grazing_areas_path': os.path.join(
            DATA_DIR, 'sfu_per_soum_desert_masked.shp'),
        # 'pft_initial_table': "C:/Users/ginge/Dropbox/sample_inputs/pft_initial_table.csv",
        # 'site_initial_table': "C:/Users/ginge/Dropbox/sample_inputs/site_initial_table.csv",
        'initial_conditions_dir': os.path.join(DATA_DIR, 'initial_conditions'),
        'animal_density': os.path.join(DATA_DIR, 'animal_per_ha_winter_camp_scaled.tif'),
    }
    return args


def raster_values_at_points(
        point_shp_path, raster_path, band, raster_field_name):
    """Collect values from a raster intersecting points in a shapefile.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates and must have a site id field called 'site_id'
        raster_path (string): path to raster containing values that should be
            extracted at points
        band (int): band index of the raster to analyze
        raster_field_name (string): name to assign to the field in the data
            frame that contains values extracted from the raster

    Returns:
        a pandas data frame with one column 'site_id' containing site_id values
            of point features, and one column raster_field_name containing
            values from the raster at the point location

    """
    raster_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    point_vector = ogr.Open(point_shp_path)
    point_layer = point_vector.GetLayer()

    # build up a list of the original field names so we can copy it to report
    point_field_name_list = ['site_id']

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


def rpm_results_to_table(workspace_dir, n_months, point_shp_path, save_as):
    """Extract values from RPM results rasters at points and format as a table.

    For the points in `points_shp_path`, extract live and standing dead biomass
    from state variable rasters created by RPM for each month that RPM was run.

    Parameters:
        workspace_dir (string): directory where RPM results have been stored
        n_months (int): number of model steps to collect outputs from
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        save_as (string): path to location where the summary table should be
            written

    Side effects:
        creates or modifies a csv table containing live and standing dead
            biomass at the location `save_as`

    Returns:
        None

    """
    df_list = []
    for month_index in range(n_months):
        aglivc_raster_path = os.path.join(
            workspace_dir, 'state_variables_m{}'.format(month_index),
            'aglivc_1.tif')
        aglivc_df = raster_values_at_points(
            point_shp_path, aglivc_raster_path, 1, 'aglivc')

        stdedc_raster_path = os.path.join(
            workspace_dir, 'state_variables_m{}'.format(month_index),
            'stdedc_1.tif')
        stdedc_df = raster_values_at_points(
            point_shp_path, stdedc_raster_path, 1, 'stdedc')

        merged_df = aglivc_df.merge(
            stdedc_df, on='site_id', suffixes=(False, False),
            validate="one_to_one")
        merged_df['step'] = month_index
        df_list.append(merged_df)
    sum_df = pandas.concat(df_list)
    sum_df['aboveground_live_biomass_gm2'] = sum_df.aglivc * 2.5
    sum_df['standing_dead_biomass_gm2'] = sum_df.stdedc * 2.5
    sum_df['total_biomass_gm2'] = (
        sum_df.aboveground_live_biomass_gm2 + sum_df.standing_dead_biomass_gm2)
    sum_df.to_csv(save_as, index=False)


def copy_rpm_outputs(workspace_dir, n_months, copy_dir):
    """Copy aglivc and stdedc outputs to `copy_dir`."""
    if not os.path.exists(copy_dir):
        os.makedirs(copy_dir)

    for month_index in range(n_months):
        aglivc_raster_path = os.path.join(
            workspace_dir, 'state_variables_m{}'.format(month_index),
            'aglivc_1.tif')
        output_path = os.path.join(
            copy_dir, 'aglivc_m{}.tif'.format(month_index))
        shutil.copyfile(aglivc_raster_path, output_path)

        stdedc_raster_path = os.path.join(
            workspace_dir, 'state_variables_m{}'.format(month_index),
            'stdedc_1.tif')
        output_path = os.path.join(
            copy_dir, 'stdedc_m{}.tif'.format(month_index))
        shutil.copyfile(stdedc_raster_path, output_path)


def extend():
    """Launch the model to extend from previous results."""
    # the final step of the previous model run. This becomes m-1
    last_successful_step = 25
    # total number of months to run
    total_n_months = 39
    n_months = total_n_months - last_successful_step
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/temp_model_outputs_initialized_by_Century"
    existing_dir = os.path.join(outer_dir, 'uniform_density_per_soum')
    extend_dir = os.path.join(outer_dir, 'uniform_density_per_soum_extend')
    if not os.path.exists(extend_dir):
        os.makedirs(extend_dir)
    rpm_args = generate_base_args(extend_dir)

    # copy initial conditions from ending state of existing runs
    initial_conditions_dir = tempfile.mkdtemp()
    required_bn_list = [
        f for f in os.listdir(os.path.join(
            existing_dir, 'state_variables_m-1')) if f.endswith('.tif')]
    existing_bn_list = [
        f for f in os.listdir(os.path.join(
            existing_dir, 'state_variables_m{}'.format(last_successful_step)))
        if f.endswith('.tif')]
    missing_bn_list = set(required_bn_list).difference(existing_bn_list)
    source_path_list = (
        [os.path.join(existing_dir, 'state_variables_m{}'.format(
            last_successful_step), e_bn) for e_bn in existing_bn_list] +
        [os.path.join(existing_dir, 'state_variables_m-1', m_bn) for m_bn in
        missing_bn_list])
    for path in source_path_list:
        shutil.copyfile(
            path, os.path.join(initial_conditions_dir, os.path.basename(path)))

    # reset args to start from original simulation results
    rpm_args['initial_conditions_dir'] = initial_conditions_dir
    rpm_args['n_months'] = n_months
    rpm_args['starting_year'] = 2018
    rpm_args['starting_month'] = 11
    forage.execute(rpm_args)

    # copy outputs from extend simulation into original
    for extend_index in range(n_months):
        src_dir = os.path.join(
            extend_dir, 'state_variables_m{}'.format(extend_index))
        new_index = extend_index + last_successful_step + 1
        dest_dir = os.path.join(
            existing_dir, 'state_variables_m{}'.format(new_index))
        shutil.copytree(src_dir, dest_dir)


def CHIRPS_precip_to_table(
        n_months, starting_month, starting_year, point_shp_path, save_as):
    """Extract values from CHIRPS rasters and write to table.

    Parameters:
        n_months (int): number of model steps to collect outputs from
        starting_month (int): starting month to collect (i.e. 1=January)
        starting_year (int): starting year to collect
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        save_as (string): path to location where the summary table should be
            written

    Side effects:
        creates or modifies a csv table containing monthly precipitation at the
        location `save_as`

    Returns:
        None

    """
    chirps_path_pattern = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/CHIRPS_div_by_10/chirps-v2.0.<year>.<month>.tif"
    df_list = []
    for month_index in range(n_months):
        current_month = (starting_month + month_index - 1) % 12 + 1
        current_year = starting_year + (starting_month + month_index - 1) // 12
        precip_path = chirps_path_pattern.replace(
            '<year>', str(current_year)).replace(
                '<month>', '{:02d}'.format(current_month))
        precip_df = raster_values_at_points(
            point_shp_path, precip_path, 1, 'precip_cm')
        precip_df['step'] = month_index
        df_list.append(precip_df)
    sum_df = pandas.concat(df_list)
    sum_df.to_csv(save_as, index=False)


def animal_density_to_table(
        workspace_dir, n_months, starting_month, starting_year, point_shp_path,
        save_as):
    """Extract animal density values at points and write to table.

    Parameters:
        workspace_dir (string): directory where RPM results have been stored
        n_months (int): number of model steps to collect outputs from
        starting_month (int): starting month to collect (i.e. 1=January)
        starting_year (int): starting year to collect
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        save_as (string): path to location where the summary table should be
            written

    Side effects:
        creates or modifies a csv table containing animal density at the
        location `save_as`

    Returns:
        None

    """
    density_path_pattern = os.path.join(
        workspace_dir, 'output', 'animal_density_<year>_<month>.tif')
    df_list = []
    for month_index in range(n_months):
        current_month = (starting_month + month_index - 1) % 12 + 1
        current_year = starting_year + (starting_month + month_index - 1) // 12
        density_path = density_path_pattern.replace(
            '<year>', str(current_year)).replace('<month>', str(current_month))
        density_df = raster_values_at_points(
            point_shp_path, density_path, 1, 'sfu_per_ha')
        density_df['step'] = month_index
        df_list.append(density_df)
    sum_df = pandas.concat(df_list)
    sum_df.to_csv(save_as, index=False)


def collect_precip_and_animal_density():
    """wrapper to extract precip and animal density values to points."""
    precip_table_dir = "C:/Users/ginge/Desktop/CHIRPS_precip_tables"
    if not os.path.exists(precip_table_dir):
        os.makedirs(precip_table_dir)
    n_months = 39
    starting_month = 9
    starting_year = 2016
    for point_shp_path in SAMPLE_PATH_LIST:
        save_as = os.path.join(
            precip_table_dir, "CHIRPS_precip_{}.csv".format(
                os.path.basename(point_shp_path)[:-4]))
        CHIRPS_precip_to_table(
            n_months, starting_month, starting_year, point_shp_path, save_as)

    rpm_results_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/temp_model_outputs_initialized_by_Century/compare_to_ndvi_v2"
    animal_density_dir = "C:/Users/ginge/Desktop/animal_density_tables"
    if not os.path.exists(animal_density_dir):
        os.makedirs(animal_density_dir)
    n_months = 16
    for point_shp_path in SAMPLE_PATH_LIST:
        save_as = os.path.join(
            animal_density_dir, "animal_density_{}.csv".format(
                os.path.basename(point_shp_path)[:-4]))
        animal_density_to_table(
            rpm_results_dir, n_months, starting_month, starting_year,
            point_shp_path, save_as)


def main():
    """Launch model and check results."""
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/temp_model_outputs_initialized_by_Century"
    workspace_dir = os.path.join(outer_dir, 'winter_camps_v3')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    rpm_args = generate_base_args(workspace_dir)
    # TODO remove
    forage.reclassify_nodata(rpm_args['animal_density'], -1.0)
    forage.execute(rpm_args)

    # extract results to a table
    for point_shp_path in SAMPLE_PATH_LIST:
        n_months = 39  # TODO how many ran successfully?
        save_as = "C:/Users/ginge/Desktop/winter_camps_v3/biomass_summary_{}.csv".format(
            os.path.basename(point_shp_path)[:-4])
        rpm_results_to_table(workspace_dir, n_months, point_shp_path, save_as)

    # copy aglivc and stdedc rasters to a folder to upload and share
    copy_dir = "C:/Users/ginge/Desktop/winter_camps_v3/RPM_outputs"
    copy_rpm_outputs(workspace_dir, n_months, copy_dir)


if __name__ == "__main__":
    __spec__ = None  # for running with pdb
    # extend()
    main()
