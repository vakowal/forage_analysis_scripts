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

# folder containing aggregated resampled NDVI through 2018
NDVI_DIR = "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_3day/monthly_avg_aggregated"

# global value for crude protein, Kenya and Mongolia
CRUDE_PROTEIN = 0.14734

LAIKIPIA_DATA_DIR = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs"

GLOBAL_SOIL_DIR = "E:/GIS_local_archive/General_useful_data/soilgrids1k/converted_to_Century_units"


def monitoring_area_scenario_args(workspace_dir):
    """Generate an args dict of inputs for RPM scenarios in monitoring area."""
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'aoi_path': os.path.join(
            DATA_DIR, 'soums_monitoring_area_diss.shp'),
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            DATA_DIR, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            DATA_DIR, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            DATA_DIR, 'soil', 'pH.tif'),
        'clay_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            DATA_DIR, 'soil', 'sand.tif'),
        'min_temp_path_pattern': os.path.join(
            DATA_DIR, 'temp', 'wc2.0_30s_tmin_<month>.tif'),
        'max_temp_path_pattern': os.path.join(
            DATA_DIR, 'temp', 'wcs2.0_30s_tmax_<month>.tif'),
        'monthly_vi_path_pattern': os.path.join(
            NDVI_DIR, 'ndvi_<year>_<month>.tif'),
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
        'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/SCP_sites/worldclim_historical_sch/RPM_inputs/initial_conditions",
    }
    return args


def generate_base_args(workspace_dir):
    """Generate an args dict of basic inputs for rangeland production model."""
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 9,
        'starting_year': 2016,
        'n_months': 39,  # 39 to run all the way through 2019
        'aoi_path': os.path.join(
            DATA_DIR, 'soums_monitoring_area_diss.shp'),
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            DATA_DIR, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            DATA_DIR, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            DATA_DIR, 'soil', 'pH.tif'),
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
            NDVI_DIR, 'ndvi_<year>_<month>.tif'),
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
    }
    return args


def ahlborn_args(workspace_dir):
    """Generate an args dict of inputs for rangeland production model."""
    inputs_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs"
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 9,
        'starting_year': 2011,
        'n_months': 52,
        'aoi_path': os.path.join(
            inputs_dir, "sampling_points_aoi.shp"),
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            inputs_dir, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            inputs_dir, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            inputs_dir, 'soil', 'ph.tif'),
        'clay_proportion_path': os.path.join(
            inputs_dir, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            inputs_dir, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            inputs_dir, 'soil', 'sand.tif'),
        'precip_dir': "E:/GIS_local_archive/General_useful_data/CHIRPS_cm",
        'min_temp_dir': (
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_min"),
        'max_temp_dir': (
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_max"),
        'monthly_vi_path_pattern': (
            "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_Ahlborn_sites/exported_to_tif/ndvi_<year>_<month>.tif"),
        'site_param_table': os.path.join(
            inputs_dir, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            inputs_dir, 'site_index_1.tif'),
        'veg_trait_path': os.path.join(inputs_dir, 'pft_trait.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            inputs_dir, 'pft_<PFT>.tif'),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/sfu.csv",
        'animal_grazing_areas_path': os.path.join(
            inputs_dir, "sampling_points_aoi.shp"),
        # 'pft_initial_table': "C:/Users/ginge/Dropbox/sample_inputs/pft_initial_table.csv",
        # 'site_initial_table': "C:/Users/ginge/Dropbox/sample_inputs/site_initial_table.csv",
        'initial_conditions_dir': os.path.join(
            inputs_dir, 'initial_conditions_Aug20'),
    }
    return args


def ahlborn_scenario_args(workspace_dir):
    """Arguments to run RPM for scenarios at Ahlborn sites."""
    inputs_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs"
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'aoi_path': os.path.join(
            inputs_dir, "sampling_points_aoi.shp"),
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            inputs_dir, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            inputs_dir, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            inputs_dir, 'soil', 'ph.tif'),
        'clay_proportion_path': os.path.join(
            inputs_dir, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            inputs_dir, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            inputs_dir, 'soil', 'sand.tif'),
        'min_temp_dir': (
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_min"),
        'max_temp_dir': (
            "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_max"),
        'monthly_vi_path_pattern': (
            "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_Ahlborn_sites/exported_to_tif/ndvi_<year>_<month>.tif"),
        'site_param_table': os.path.join(
            inputs_dir, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            inputs_dir, 'site_index_1.tif'),
        'veg_trait_path': os.path.join(inputs_dir, 'pft_trait.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            inputs_dir, 'pft_<PFT>.tif'),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/sfu.csv",
        'animal_grazing_areas_path': os.path.join(
            inputs_dir, "sampling_points_aoi.shp"),
        'initial_conditions_dir': os.path.join(
            inputs_dir, 'initial_conditions_scenarios'),
        'crude_protein': CRUDE_PROTEIN,
    }
    return args

def map_FID_to_field(shp_path, field):
    """Map FID of each feature, according to GetFID(), to the given field.

    This allows for mapping of a dictionary of zonal statistics, where keys
    correspond to FID according to GetFID(), to another field that is preferred
    to identify features.

    Parameters:
        shp_path (string): path to shapefile
        field (string): the field to map to FID

    Returns:
        dictionary indexed by the FID of each feature retrieved with GetFID(),
            and values are the value of `field` for the feature

    """
    vector = gdal.OpenEx(shp_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    FID_to_field = {
        feature.GetFID(): feature.GetField(field) for feature in layer}

    # clean up
    vector = None
    layer = None
    return FID_to_field


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


def rpm_outputs_to_table(
        workspace_dir, n_months, starting_year, starting_month, point_shp_path,
        shp_id_field, save_as):
    """Extract values from RPM output rasters at points.

    For the points in `points_shp_path`, extract total standing biomass and
    animal density from output rasters created by RPM for each month that RPM
    was run.

    Parameters:
        workspace_dir (string): directory where RPM results have been stored
        n_months (int): number of model steps to collect outputs from
        starting_year (int): first year of the RPM run
        starting_month (int): first month of the RPM run
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        shp_id_field (string): site label, included as a field in
            `point_shp_path`
        save_as (string): path to location where the summary table should be
            written

    Side effects:
        creates or modifies a csv table containing total standing biomass and
            animal density at the location `save_as`

    Returns:
        None

    """
    df_list = []
    for month_index in range(n_months):
        month_i = (starting_month + month_index - 1) % 12 + 1
        year = starting_year + (starting_month + month_index - 1) // 12
        standing_biomass_path = os.path.join(
            workspace_dir, 'output', 'standing_biomass_{}_{}.tif'.format(
                year, month_i))
        standing_biomass_df = raster_values_at_points(
            point_shp_path, shp_id_field, standing_biomass_path, 1,
            'standing_biomass')
        # merged_df = standing_biomass_df
        animal_density_path = os.path.join(
            workspace_dir, 'output', 'animal_density_{}_{}.tif'.format(
                year, month_i))
        animal_density_df = raster_values_at_points(
            point_shp_path, shp_id_field, animal_density_path, 1,
            'animal_density')

        merged_df = standing_biomass_df.merge(
            animal_density_df, on=shp_id_field, suffixes=(False, False),
            validate="one_to_one")
        merged_df['step'] = month_index
        df_list.append(merged_df)
    sum_df = pandas.concat(df_list)
    sum_df['total_biomass_gm2'] = sum_df.standing_biomass / 10
    sum_df.to_csv(save_as, index=False)


def rpm_results_to_table(
        workspace_dir, n_months, point_shp_path, shp_id_field, save_as):
    """Extract values from RPM state variable rasters at points.

    For the points in `points_shp_path`, extract live and standing dead biomass
    from state variable rasters created by RPM for each month that RPM was run.

    Parameters:
        workspace_dir (string): directory where RPM results have been stored
        n_months (int): number of model steps to collect outputs from
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        shp_id_field (string): site label, included as a field in
            `point_shp_path`
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
            point_shp_path, shp_id_field, aglivc_raster_path, 1, 'aglivc')

        stdedc_raster_path = os.path.join(
            workspace_dir, 'state_variables_m{}'.format(month_index),
            'stdedc_1.tif')
        stdedc_df = raster_values_at_points(
            point_shp_path, shp_id_field, stdedc_raster_path, 1, 'stdedc')

        merged_df = aglivc_df.merge(
            stdedc_df, on=shp_id_field, suffixes=(False, False),
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
        chirps_path_pattern, n_months, starting_month, starting_year,
        point_shp_path, shp_id_field, save_as):
    """Extract values from CHIRPS rasters and write to table.

    Parameters:
        chirps_path_pattern (string): template path to CHIRPS raster file
            location, where <year> and <month> can be replaced by the year and
            month. This is an argument to RPM.
        n_months (int): number of model steps to collect outputs from
        starting_month (int): starting month to collect (i.e. 1=January)
        starting_year (int): starting year to collect
        point_shp_path (string): path to shapefile containing points where
            values should be collected. must be in geographic coordinates
        shp_id_field (string): field in point_shp_path identifying features
        save_as (string): path to location where the summary table should be
            written

    Side effects:
        creates or modifies a csv table containing monthly precipitation at the
        location `save_as`

    Returns:
        None

    """
    df_list = []
    for month_index in range(n_months):
        current_month = (starting_month + month_index - 1) % 12 + 1
        current_year = starting_year + (starting_month + month_index - 1) // 12
        precip_path = chirps_path_pattern.replace(
            '<year>', str(current_year)).replace(
                '<month>', '{:02d}'.format(current_month))
        precip_df = raster_values_at_points(
            point_shp_path, shp_id_field, precip_path, 1, 'precip_cm')
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


def wcs_monitoring_sites_workflow():
    """Launch model for WCS monitoring sites study area."""
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/temp_model_outputs_initialized_by_Century"
    workspace_dir = os.path.join(
        outer_dir, 'compare_to_normalized_ndvi_doubled_6.9.20')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    rpm_args = generate_base_args(workspace_dir)
    rpm_args['save_sv_rasters'] = False
    forage.execute(rpm_args)

    # extract results to a table
    # for version in ['compare_to_normalized_ndvi_doubled']:
    # # #        'winter_camps_doubled', 'uniform_density_per_soum_doubled',
    #     workspace_dir = os.path.join(outer_dir, version)
    #     table_dir = os.path.join("C:/Users/ginge/Desktop/", version)
    #     if not os.path.exists(table_dir):
    #         os.makedirs(table_dir)
    #     for point_shp_path in SAMPLE_PATH_LIST:
    #         n_months = 39
    #         save_as = os.path.join(
    #             table_dir, "biomass_summary_{}.csv".format(
    #                 os.path.basename(point_shp_path)[:-4]))
    #         rpm_results_to_table(
    #             workspace_dir, n_months, point_shp_path, save_as)
    # extract animal density to a table
    # for version in ['compare_to_ndvi_v3', 'compare_to_ndvi_v3_doubled']:
    #     workspace_dir = os.path.join(outer_dir, version)
    #     table_dir = os.path.join("C:/Users/ginge/Desktop/", version)
    #     if not os.path.exists(table_dir):
    #         os.makedirs(table_dir)
    #     for point_shp_path in SAMPLE_PATH_LIST:
    #         n_months = 39
    #         starting_month = rpm_args['starting_month']
    #         starting_year = rpm_args['starting_year']
    #         save_as = os.path.join(
    #             table_dir, "animal_density_{}.csv".format(
    #                 os.path.basename(point_shp_path)[:-4]))
    #         animal_density_to_table(
    #             workspace_dir, n_months, starting_month, starting_year,
    #             point_shp_path, save_as)

    # copy aglivc and stdedc rasters to a folder to upload and share
    copy_dir = "C:/Users/ginge/Desktop/zero_sd_v2/RPM_outputs"
    # copy_rpm_outputs(workspace_dir, n_months, copy_dir)


def extract_exclosure_results():
    """Extract tabular data from model runs at exclosure points."""
    exclosure_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/summaries_GK/exclosure_data_2017_2018_2019.shp"
    table_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_results/WCS_exclosures/RPM_initialized_by_Century"
    model_output_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/temp_model_outputs_initialized_by_Century"
    n_months = 39
    for version in ['uniform_density_per_soum', 'zero_sd_v2']:
        workspace_dir = os.path.join(model_output_dir, version)
        save_as = os.path.join(
            table_dir, 'RPM_biomass_exclosure_points_{}.csv'.format(version))
        rpm_results_to_table(
            workspace_dir, n_months, exclosure_shp_path, save_as)


def ahlborn_sites_workflow():
    """Launch the model for Julian Ahlborn's study area."""
    # outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_sites_RPM_outputs/zero_sd"
    # outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_sites_RPM_outputs/crude_protein_enforced"
    # outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_sites_RPM_outputs/via_NDVI_TEST"
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_sites_RPM_outputs/uniform_density_sfu_per_site"
    aoi_pattern = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_centroid_buffer_sfu/aoi_<idx>.shp"
    for aoi_idx in range(1, 16):
        workspace_dir = os.path.join(outer_dir, 'aoi_{}'.format(aoi_idx))
        rpm_args = ahlborn_args(workspace_dir)
        aoi_path = aoi_pattern.replace('<idx>', str(aoi_idx))
        rpm_args['aoi_path'] = aoi_path
        rpm_args['animal_grazing_areas_path'] = aoi_path
        rpm_args['animal_density'] = "C:/Users/ginge/Desktop/sfu_lowres/sfu_per_ha_est.tif"
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
            forage.execute(rpm_args)

    # simulated biomass at centroid of each plot (distance treatment)
    ahlborn_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_sites_RPM_outputs"
    # point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/sampling_sites_shapefile/plot_centroids.shp"

    # simulated biomass at site centroids
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/sampling_sites_shapefile/site_centroids.shp"
    for run_id in ['uniform_density_sfu_per_site']:
    #         'uniform_density_sfu_per_site','zero_sd', 'compare_to_ndvi']:
        n_months = rpm_args['n_months']
        outer_dir = os.path.join(ahlborn_dir, run_id)
        intermediate_dir = os.path.join(
            outer_dir, 'biomass_tables_intermediate')
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        df_path_list = []
        for aoi_idx in range(1, 16):
            workspace_dir = os.path.join(outer_dir, 'aoi_{}'.format(aoi_idx))
            target_table_path = os.path.join(
                intermediate_dir,
                'RPM_biomass_plot_centroids_aoi_{}.csv'.format(aoi_idx))
            rpm_outputs_to_table(
                workspace_dir, n_months, rpm_args['starting_year'],
                rpm_args['starting_month'], point_shp_path, 'site',
                target_table_path)
            df_path_list.append(target_table_path)
        # merge tables collected from the separate aoi into one table
        reduced_df_list = []
        df_i = 0
        while df_i < len(df_path_list):
            full_df = pandas.read_csv(df_path_list[df_i])
            reduced_df = full_df.dropna()
            reduced_df_list.append(reduced_df)
            df_i = df_i + 1
        shutil.rmtree(intermediate_dir)
        combined_df = pandas.concat(reduced_df_list)
        full_table_path = os.path.join(
            outer_dir,
            'RPM_biomass_site_centroids_{}.csv'.format(run_id))
        combined_df.to_csv(full_table_path)

    # animal density at each plot, for NDVI runs
    # run_id = 'via_ndvi'
    # rpm_args = ahlborn_args(workspace_dir)
    # n_months = rpm_args['n_months']
    # outer_dir = os.path.join(ahlborn_dir, run_id)
    # intermediate_dir = os.path.join(
    #     outer_dir, 'density_tables_intermediate')
    # if not os.path.exists(intermediate_dir):
    #     os.makedirs(intermediate_dir)
    # df_path_list = []
    # for aoi_idx in range(1, 16):
    #     workspace_dir = os.path.join(outer_dir, 'aoi_{}'.format(aoi_idx))
    #     target_table_path = os.path.join(
    #         intermediate_dir,
    #         'RPM_density_plot_centroids_aoi_{}.csv'.format(aoi_idx))
    #     starting_year = 2011
    #     starting_month = 9
    #     rpm_outputs_to_table(
    #         workspace_dir, n_months, starting_year, starting_month,
    #         point_shp_path, 'plotid', target_table_path)
    #     df_path_list.append(target_table_path)
    # reduced_df_list = []
    # df_i = 0
    # while df_i < len(df_path_list):
    #     full_df = pandas.read_csv(df_path_list[df_i])
    #     reduced_df = full_df.dropna()
    #     reduced_df_list.append(reduced_df)
    #     df_i = df_i + 1
    # combined_df = pandas.concat(reduced_df_list)
    # full_table_path = os.path.join(
    #     outer_dir, 'RPM_density_plot_centroids_via_ndvi.csv')
    # combined_df.to_csv(full_table_path)


def ahlborn_sites_collect_precip():
    """Make a table of precipitation at Julian's sites from CHIRPS."""
    rpm_args = ahlborn_args("C:/Users/ginge/Desktop")
    chirps_path_pattern = rpm_args['monthly_precip_path_pattern']
    n_months = rpm_args['n_months']
    starting_month = rpm_args['starting_month']
    starting_year = rpm_args['starting_year']
    point_shp_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/sampling_sites_shapefile/plot_centroids.shp"
    save_as = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/Julian_Ahlborn/CHIRPS_precip_plot_centroids.csv"
    CHIRPS_precip_to_table(
        chirps_path_pattern, n_months, starting_month, starting_year,
        point_shp_path, 'plotid', save_as)


def gobi_scenarios():
    baseline_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/baseline/wc2.0_30s_prec_<year>_<month>.tif"
    high_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/multiplied_by_1.6/wc2.0_30s_prec_<year>_<month>.tif"
    low_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/multiplied_by_0.4/wc2.0_30s_prec_<year>_<month>.tif"

    baseline_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/animal_per_ha_uniform_soum.tif"
    # high_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/animal_per_ha_uniform_soum_x1.6.tif"
    high_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/animal_per_ha_uniform_soum_doubled.tif"
    zero_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century/zero_animals.tif"
    scenarios_dict = {
        'A': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'B': {
            'monthly_precip_path_pattern': high_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'C': {
            'monthly_precip_path_pattern': low_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'D': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': high_animal_density,
        },
        'F': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': zero_animal_density,
        },
        'G': {
            'monthly_precip_path_pattern': high_precip_pattern,
            'animal_density_path': high_animal_density,
        },
        'J': {
            'monthly_precip_path_pattern': low_precip_pattern,
            'animal_density_path': high_animal_density,
        },
    }
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/RPM_scenarios/revised_9.8.20"
    for scenario in scenarios_dict:
        workspace_dir = os.path.join(outer_dir, scenario)
        rpm_args = monitoring_area_scenario_args(workspace_dir)
        rpm_args['monthly_precip_path_pattern'] = scenarios_dict[
            scenario]['monthly_precip_path_pattern']
        rpm_args['animal_density'] = scenarios_dict[
            scenario]['animal_density_path']
        forage.execute(rpm_args)


def mask_and_perturb(
        base_raster_path_band, mask_vector_path, target_mask_raster_path,
        perc_change, multiplier=None):
    """Mask a raster band with a given vector and perturb by percent change.

    This was copied from pygeoprocessing.mask_raster() except that the masked
    raster may also be multiplied by a constant value. If a multiplier is
    supplied, the data type of the target raster is float; if not, it is the
    same as the datatype of the input raster.

    Args:
        base_raster_path_band (tuple): a (path, band number) tuple indicating
            the data to mask.
        mask_vector_path (path): path to a vector that will be used to mask
            anything outside of the polygon that overlaps with
            ``base_raster_path_band`` to ``target_mask_value`` if defined or
            else ``base_raster_path_band``'s nodata value.
        target_mask_raster_path (str): path to desired target raster that
            is a copy of ``base_raster_path_band`` except any pixels that do
            not intersect with ``mask_vector_path`` are set to
            ``target_mask_value`` or ``base_raster_path_band``'s nodata value
            if ``target_mask_value`` is None.
        perc_change (float): percent change that should be applied to the
            masked raster
        multiplier (float or int): if supplied, the masked raster is multiplied
            by this value before being perturbed by percent change. For
            example, to convert between units.

    Returns:
        None

    """
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTIFF', (
        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
        'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
    with tempfile.NamedTemporaryFile(
            prefix='mask_raster', delete=False,
            suffix='.tif') as mask_raster_file:
        mask_raster_path = mask_raster_file.name

    pygeoprocessing.new_raster_from_base(
        base_raster_path_band[0], mask_raster_path, gdal.GDT_Byte, [255],
        fill_value_list=[0],
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)

    base_raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    base_nodata = base_raster_info['nodata'][base_raster_path_band[1]-1]

    pygeoprocessing.rasterize(
        mask_vector_path, mask_raster_path, burn_values=[1],
        layer_id=0, option_list=[('ALL_TOUCHED=FALSE')])

    def mask_op(base_array, mask_array, perc_change, multiplier):
        valid_mask = (~numpy.isclose(base_array, base_nodata))
        converted = numpy.copy(base_array) * multiplier
        result = numpy.empty(base_array.shape, dtype=numpy.float32)
        result[:] = base_nodata
        result[valid_mask] = (
            converted[valid_mask] + abs(converted[valid_mask]) * perc_change)
        result[mask_array == 0] = base_nodata
        return result

    if multiplier is None:
        multiplier = 1
        target_datatype = base_raster_info['datatype']
    else:
        target_datatype = gdal.GDT_Float32

    pygeoprocessing.raster_calculator(
        [base_raster_path_band, (mask_raster_path, 1), (perc_change, 'raw'),
        (multiplier, 'raw')],
        mask_op, target_mask_raster_path, target_datatype, base_nodata,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)

    os.remove(mask_raster_path)


def Ahlborn_scenarios():
    outer_aoi_path = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/sampling_points_aoi.shp"
    worldclim_precip_dir = "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_precip"
    baseline_tmin_dir = "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_min"
    baseline_tmax_dir = "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_max"

    # multiplier to convert Worldclim precip (mm) to cm
    precip_multiplier = 0.1

    # percent change in precip, tmin, tmax derived from future GCM scenarios
    # 1.0 = 100%
    precip_perc_change = 0.22
    tmin_perc_change = 1.72
    tmax_perc_change = 0.71

    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Ahlborn_scenarios/Dec102020"

    # create temporary climate directories, perturb climate inputs by constant
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    # temp_dir = tempfile.mkdtemp(dir=outer_dir)
    temp_dir = os.path.join(outer_dir, 'tmpdqf19rxa')
    baseline_precip_dir = os.path.join(temp_dir, 'precip_baseline')
    elevated_precip_dir = os.path.join(temp_dir, 'precip_elevated')
    elevated_tmin_dir = os.path.join(temp_dir, 'tmin_elevated')
    elevated_tmax_dir = os.path.join(temp_dir, 'tmax_elevated')
    # os.makedirs(baseline_precip_dir)
    # os.makedirs(elevated_precip_dir)
    # os.makedirs(elevated_tmin_dir)
    # os.makedirs(elevated_tmax_dir)

    for m in range(1, 13):
        input_precip_path = os.path.join(
            worldclim_precip_dir, 'wc2.0_30s_prec_{:02}.tif'.format(m))
        baseline_precip_path = os.path.join(
            baseline_precip_dir, 'precip_2016_{}.tif'.format(m))
        mask_and_perturb(
            (input_precip_path, 1), outer_aoi_path, baseline_precip_path,
            perc_change=0, multiplier=precip_multiplier)
        shutil.copyfile(
            baseline_precip_path,
            os.path.join(baseline_precip_dir, 'precip_2017_{}.tif'.format(m)))

        elevated_precip_path = os.path.join(
            elevated_precip_dir, 'precip_2016_{}.tif'.format(m))
        mask_and_perturb(
            (input_precip_path, 1), outer_aoi_path, elevated_precip_path,
            perc_change=precip_perc_change, multiplier=precip_multiplier)
        shutil.copyfile(
            elevated_precip_path,
            os.path.join(elevated_precip_dir, 'precip_2017_{}.tif'.format(m)))

        input_tmin_path = os.path.join(
            baseline_tmin_dir, 'wc2.0_30s_tmin_{}.tif'.format(m))
        elevated_tmin_path = os.path.join(
            elevated_tmin_dir, 'tmin_{}.tif'.format(m))
        mask_and_perturb(
            (input_tmin_path, 1), outer_aoi_path, elevated_tmin_path,
            perc_change=tmin_perc_change)

        input_tmax_path = os.path.join(
            baseline_tmax_dir, 'wc2.0_30s_tmax_{}.tif'.format(m))
        elevated_tmax_path = os.path.join(
            elevated_tmax_dir, 'tmax_{}.tif'.format(m))
        mask_and_perturb(
            (input_tmax_path, 1), outer_aoi_path, elevated_tmax_path,
            perc_change=tmax_perc_change)

    baseline_animal_density = "C:/Users/ginge/Desktop/sfu_lowres/sfu_per_ha_est.tif"
    elevated_animal_density = "C:/Users/ginge/Desktop/sfu_lowres/sfu_per_ha_est_doubled.tif"
    zero_animal_density = "C:/Users/ginge/Desktop/sfu_lowres/zero_animals.tif"
    scenarios_dict = {
        'B': {
            'precip_dir': baseline_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': baseline_animal_density,
        },
        'C': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': baseline_animal_density,
        },
        'D': {
            'precip_dir': baseline_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': elevated_animal_density,
        },
        'E': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': elevated_animal_density,
        },
        'F': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': baseline_tmin_dir,
            'tmax_dir': baseline_tmax_dir,
            'animal_density_path': baseline_animal_density,
        },
        'G': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': baseline_tmin_dir,
            'tmax_dir': baseline_tmax_dir,
            'animal_density_path': elevated_animal_density,
        },
        'H': {
            'precip_dir': baseline_precip_dir,
            'tmin_dir': baseline_tmin_dir,
            'tmax_dir': baseline_tmax_dir,
            'animal_density_path': elevated_animal_density,
        },
        'I': {
            'precip_dir': baseline_precip_dir,
            'tmin_dir': baseline_tmin_dir,
            'tmax_dir': baseline_tmax_dir,
            'animal_density_path': zero_animal_density,
        },
        'J': {
            'precip_dir': baseline_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': zero_animal_density,
        },
        'K': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': elevated_tmin_dir,
            'tmax_dir': elevated_tmax_dir,
            'animal_density_path': zero_animal_density,
        },
        'L': {
            'precip_dir': elevated_precip_dir,
            'tmin_dir': baseline_tmin_dir,
            'tmax_dir': baseline_tmax_dir,
            'animal_density_path': zero_animal_density,
        },
    }
    aoi_pattern = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs/site_centroid_buffer_sfu/aoi_<idx>.shp"
    for aoi_idx in range(1, 16):
        for scenario in scenarios_dict:
            workspace_dir = os.path.join(
                outer_dir, 'aoi_{}'.format(aoi_idx), scenario)
            if not os.path.exists(workspace_dir):
                os.makedirs(workspace_dir)
                rpm_args = ahlborn_scenario_args(workspace_dir)
                aoi_path = aoi_pattern.replace('<idx>', str(aoi_idx))
                rpm_args['n_months'] = 24
                rpm_args['aoi_path'] = aoi_path
                rpm_args['precip_dir'] = scenarios_dict[
                    scenario]['precip_dir']
                rpm_args['min_temp_dir'] = scenarios_dict[
                    scenario]['tmin_dir']
                rpm_args['max_temp_dir'] = scenarios_dict[
                    scenario]['tmax_dir']
                rpm_args['animal_density'] = scenarios_dict[
                    scenario]['animal_density_path']
                forage.execute(rpm_args)

    # clean up
    # shutil.rmtree(baseline_precip_dir)
    # shutil.rmtree(elevated_precip_dir)
    # shutil.rmtree(elevated_tmin_dir)
    # shutil.rmtree(elevated_tmax_dir)


def summarize_outputs_at_OPC_transects(workspace_dir):
    """Collect RPM outputs at transect locations on OPC."""
    transect_shp_path = "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/OPC_transect_RPM_date.shp"
    # date of transect collection, corrected to align with end of the month
    year_month_list = [
        "2015_3", "2015_10", "2015_4", "2014_10", "2015_5", "2014_11",
        "2015_9", "2015_6", "2015_2", "2015_11"]
    df_list = []
    for date in year_month_list:
        animal_density_path = os.path.join(
            workspace_dir, 'output', 'animal_density_{}.tif'.format(date))
        animal_density_df = raster_values_at_points(
            transect_shp_path, 'transect', animal_density_path, 1,
            'animal_density')
        biomass_path = os.path.join(
            workspace_dir, 'output', 'standing_biomass_{}.tif'.format(date))
        biomass_df = raster_values_at_points(
            transect_shp_path, 'transect', biomass_path, 1, 'standing_biomass')
        RPM_df = animal_density_df.merge(
            biomass_df, suffixes=(False, False), validate="one_to_one")
        RPM_df['year_month'] = date
        df_list.append(RPM_df)
    sum_df = pandas.concat(df_list)
    transect_date_corr_df = pandas.read_csv(
        "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/OPC_transect_RPM_date.csv")
    sum_restr_df = sum_df.merge(
        transect_date_corr_df, suffixes=(False, False), validate="one_to_one")
    sum_restr_df.drop(['Lat', 'Long'], axis='columns', inplace=True)
    sum_restr_df.to_csv(
        os.path.join(workspace_dir, 'RPM_outputs_transect_locations.csv'),
        index=False)


def launch_RPM_OPC():
    """Run RPM at OPC."""
    def rpm_args():
        rpm_args = {
            'results_suffix': "",
            'starting_month': 1,
            'starting_year': 2013,
            'n_months': 36,
            'aoi_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/OPC_buf.shp",
            'management_threshold': 300,
            'proportion_legume_path': os.path.join(
                LAIKIPIA_DATA_DIR, 'prop_legume.tif'),
            'bulk_density_path': os.path.join(
                GLOBAL_SOIL_DIR, "bldfie_m_sl3_1km_ll.tif"),
            'ph_path': os.path.join(
                GLOBAL_SOIL_DIR, 'phihox_m_sl3_1km_ll.tif'),
            'clay_proportion_path': os.path.join(
                GLOBAL_SOIL_DIR, 'clyppt_m_sl3_1m_ll.tif'),
            'silt_proportion_path': os.path.join(
                GLOBAL_SOIL_DIR, 'sltppt_m_sl3_1m_ll.tif'),
            'sand_proportion_path': os.path.join(
                GLOBAL_SOIL_DIR, 'sndppt_m_sl3_1m_ll.tif'),
            'monthly_precip_path_pattern': "E:/GIS_local/Mongolia/CHIRPS/div_by_10/chirps-v2.0.<year>.<month>.tif",
            'min_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmin/wc2.0_30s_tmin_<month>.tif",
            'max_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmax/wc2.0_30s_tmax_<month>.tif",
            'site_param_table': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/site_parameter_table.csv",
            'site_param_spatial_index_path': os.path.join(
                LAIKIPIA_DATA_DIR, 'site_index.tif'),
            'veg_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/pft_parameter_table.csv",
            'veg_spatial_composition_path_pattern': os.path.join(
                LAIKIPIA_DATA_DIR, 'pft_<PFT>.tif'),
            'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/animal_parameter_table.csv",
            'animal_grazing_areas_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/OPC_buf.shp",
            'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/RPM_inputs/initial_conditions",
            'crude_protein': CRUDE_PROTEIN,
        }
        return rpm_args

    # zero animals
    workspace_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_OPC_zero_sd"
    model_args = rpm_args()
    model_args['animal_density'] = os.path.join(
        LAIKIPIA_DATA_DIR, "zero_animals.tif")
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        model_args['workspace_dir'] = workspace_dir
        forage.execute(model_args)
        summarize_outputs_at_OPC_transects(workspace_dir)

    # uniform density - doubled
    workspace_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_OPC_uniform_sd_doubled_animals"
    model_args = rpm_args()
    model_args['animal_density'] = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/OPC_RPM/cattle_equiv_per_ha_doubled.tif"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        model_args['workspace_dir'] = workspace_dir
        forage.execute(model_args)
        summarize_outputs_at_OPC_transects(workspace_dir)

    # via NDVI
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_via_NDVI_OPC"
    workspace_dir = os.path.join(
        outer_dir, 'fitted_through_2015_doubled_animals')  # TODO remove
    model_args = rpm_args()
    # TODO remove
    model_args['animal_grazing_areas_path'] = "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/OPC_buf_doubled_animals.shp"
    model_args['monthly_vi_path_pattern'] = (
        "E:/GIS_local_archive/Kenya_ticks/NDVI/fitted_3_day_average_monthly/export_to_tif/NDVI_<year>_<month>.tif")
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        model_args['workspace_dir'] = workspace_dir
        forage.execute(model_args)
        summarize_outputs_at_OPC_transects(workspace_dir)


def Laikipia_NDVI_test():
    """Test using NDVI to disaggregate animals across Laikipia county."""
    rpm_args = {
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2013,
        'n_months': 36,
        'aoi_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/Laikipia.shp",
        'management_threshold': 300,
        'proportion_legume_path': os.path.join(
            LAIKIPIA_DATA_DIR, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            GLOBAL_SOIL_DIR, "bldfie_m_sl3_1km_ll.tif"),
        'ph_path': os.path.join(
            GLOBAL_SOIL_DIR, 'phihox_m_sl3_1km_ll.tif'),
        'clay_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'clyppt_m_sl3_1m_ll.tif'),
        'silt_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'sltppt_m_sl3_1m_ll.tif'),
        'sand_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'sndppt_m_sl3_1m_ll.tif'),
        'monthly_precip_path_pattern': "E:/GIS_local/Mongolia/CHIRPS/div_by_10/chirps-v2.0.<year>.<month>.tif",
        'min_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmin/wc2.0_30s_tmin_<month>.tif",
        'max_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmax/wc2.0_30s_tmax_<month>.tif",
        'monthly_vi_path_pattern': "E:/GIS_local_archive/Kenya_ticks/NDVI/fitted_3_day_average_monthly/export_to_tif/NDVI_<year>_<month>.tif",
        'site_param_table': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/site_parameter_table.csv",
        'site_param_spatial_index_path': os.path.join(
            LAIKIPIA_DATA_DIR, 'site_index.tif'),
        'veg_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/pft_parameter_table.csv",
        'veg_spatial_composition_path_pattern': os.path.join(
            LAIKIPIA_DATA_DIR, 'pft_<PFT>.tif'),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/animal_parameter_table.csv",
        'animal_grazing_areas_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/Laikipia.shp",
        'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/initial_conditions",
        'crude_protein': CRUDE_PROTEIN,
    }
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_via_NDVI"
    workspace_dir = os.path.join(outer_dir, 'fitted_through_2015')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        rpm_args['workspace_dir'] = workspace_dir
        forage.execute(rpm_args)

    # summarize animal density
    properties_shp_path = "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/regional_properties_Jul_8_2016_as_modeled_WGS84.shp"
    animal_density_path_dict = {
        'mean_density': os.path.join(
            workspace_dir, 'output', 'summary_results',
            'mean_animal_density.tif'),
    }
    for m in range(rpm_args['n_months']):
        month = (rpm_args['starting_month'] + m - 1) % 12 + 1
        year = (
            rpm_args['starting_year'] +
            (rpm_args['starting_month'] + m - 1) // 12)
        animal_density_path_dict['{}'.format(m)] = os.path.join(
            workspace_dir, 'output', 'animal_density_{}_{}.tif'.format(
                year, month))
    df_list = []
    fid_to_name = map_FID_to_field(properties_shp_path, 'NAME')
    for column_name in animal_density_path_dict:
        zonal_stats_dict = pygeoprocessing.zonal_statistics(
            (animal_density_path_dict[column_name], 1), properties_shp_path)
        name_zonal_stats_dict = {
            name: zonal_stats_dict[fid] for (fid, name) in
            fid_to_name.items()
        }
        name_df = pandas.DataFrame(name_zonal_stats_dict)
        name_df_t = name_df.transpose()
        name_df_t['NAME'] = name_df_t.index
        name_df_t[column_name] = name_df_t['sum'] / name_df_t['count']
        name_df_t.drop(
            ['sum', 'min', 'max', 'count', 'nodata_count'], axis='columns',
            inplace=True)
        df_list.append(name_df_t)
    combined_df = df_list[0]
    df_i = 1
    while df_i < len(df_list):
        combined_df = combined_df.merge(
            df_list[df_i], on='NAME', suffixes=(False, False),
            validate="one_to_one")
        df_i = df_i + 1
    save_as = os.path.join(workspace_dir, 'animal_density_summary.csv')
    combined_df.to_csv(save_as, index=False)


def Laikipia_scenarios():
    rpm_args = {
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,  # precip rasters contain '2016' and '2017'
        'n_months': 24,
        'aoi_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/Laikipia.shp",
        'management_threshold': 300,
        'proportion_legume_path': os.path.join(
            LAIKIPIA_DATA_DIR, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            GLOBAL_SOIL_DIR, "bldfie_m_sl3_1km_ll.tif"),
        'ph_path': os.path.join(
            GLOBAL_SOIL_DIR, 'phihox_m_sl3_1km_ll.tif'),
        'clay_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'clyppt_m_sl3_1m_ll.tif'),
        'silt_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'sltppt_m_sl3_1m_ll.tif'),
        'sand_proportion_path': os.path.join(
            GLOBAL_SOIL_DIR, 'sndppt_m_sl3_1m_ll.tif'),
        'min_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmin/wc2.0_30s_tmin_<month>.tif",
        'max_temp_path_pattern': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/worldclim_tmax/wc2.0_30s_tmax_<month>.tif",
        'site_param_table': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/site_parameter_table.csv",
        'site_param_spatial_index_path': os.path.join(
            LAIKIPIA_DATA_DIR, 'site_index.tif'),
        'veg_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/pft_parameter_table.csv",
        'veg_spatial_composition_path_pattern': os.path.join(
            LAIKIPIA_DATA_DIR, 'pft_<PFT>.tif'),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/animal_parameter_table.csv",
        'animal_grazing_areas_path': "E:/GIS_local_archive/Kenya_ticks/Kenya_forage/Laikipia.shp",
        'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/initial_conditions",
        'crude_protein': CRUDE_PROTEIN,
    }
    baseline_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/baseline/wc2.0_30s_prec_<year>_<month>.tif"
    high_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/multiplied_by_1.6/wc2.0_30s_prec_<year>_<month>.tif"
    low_precip_pattern = "E:/GIS_local/Mongolia/Worldclim/multiplied_by_0.4/wc2.0_30s_prec_<year>_<month>.tif"

    baseline_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/cattle_equiv_per_ha.tif"
    high_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/cattle_equiv_per_ha_double.tif"
    zero_animal_density = "C:/Users/ginge/Dropbox/NatCap_backup/Forage_model/Forage_model/model_inputs/Laikipia_RPM/RPM_inputs/zero_animals.tif"

    scenarios_dict = {
        'A': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'B': {
            'monthly_precip_path_pattern': high_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'C': {
            'monthly_precip_path_pattern': low_precip_pattern,
            'animal_density_path': baseline_animal_density,
        },
        'D': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': high_animal_density,
        },
        'F': {
            'monthly_precip_path_pattern': baseline_precip_pattern,
            'animal_density_path': zero_animal_density,
        },
        'G': {
            'monthly_precip_path_pattern': high_precip_pattern,
            'animal_density_path': high_animal_density,
        },
        'I': {
            'monthly_precip_path_pattern': high_precip_pattern,
            'animal_density_path': zero_animal_density,
        },
        'J': {
            'monthly_precip_path_pattern': low_precip_pattern,
            'animal_density_path': high_animal_density,
        },
        'L': {
            'monthly_precip_path_pattern': low_precip_pattern,
            'animal_density_path': zero_animal_density,
        },
    }
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Kenya/RPM_scenarios"
    for scenario in scenarios_dict:
        workspace_dir = os.path.join(outer_dir, scenario)
        if not os.path.exists(workspace_dir):
            rpm_args['monthly_precip_path_pattern'] = scenarios_dict[
                scenario]['monthly_precip_path_pattern']
            rpm_args['animal_density'] = scenarios_dict[
                scenario]['animal_density_path']
            forage.execute(rpm_args)


def mask_and_divide(
        base_raster_path_band, mask_vector_path, target_mask_raster_path,
        multiplier=None, all_touched=False):
    """Mask a raster band with a given vector and divide by a constant.

    This was copied from pygeoprocessing.mask_raster() except that the masked
    raster may also be multiplied by a constant value. If a multiplier is
    supplied, the data type of the target raster is float; if not, it is the
    same as the datatype of the input raster.

    Args:
        base_raster_path_band (tuple): a (path, band number) tuple indicating
            the data to mask.
        mask_vector_path (path): path to a vector that will be used to mask
            anything outside of the polygon that overlaps with
            ``base_raster_path_band`` to ``target_mask_value`` if defined or
            else ``base_raster_path_band``'s nodata value.
        target_mask_raster_path (str): path to desired target raster that
            is a copy of ``base_raster_path_band`` except any pixels that do
            not intersect with ``mask_vector_path`` are set to
            ``target_mask_value`` or ``base_raster_path_band``'s nodata value
            if ``target_mask_value`` is None.
        multiplier (float or int): if supplied, the masked raster is multiplied
            by this value
        all_touched (bool): if False, a pixel is only masked if its centroid
            intersects with the mask. If True a pixel is masked if any point
            of the pixel intersects the polygon mask.

    Returns:
        None

    """
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTIFF', (
        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
        'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
    with tempfile.NamedTemporaryFile(
            prefix='mask_raster', delete=False,
            suffix='.tif') as mask_raster_file:
        mask_raster_path = mask_raster_file.name

    pygeoprocessing.new_raster_from_base(
        base_raster_path_band[0], mask_raster_path, gdal.GDT_Byte, [255],
        fill_value_list=[0],
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)

    base_raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    base_nodata = base_raster_info['nodata'][base_raster_path_band[1]-1]

    pygeoprocessing.rasterize(
        mask_vector_path, mask_raster_path, burn_values=[1],
        layer_id=0, option_list=[('ALL_TOUCHED=%s' % all_touched).upper()])

    def mask_op(base_array, mask_array, multiplier):
        result = numpy.copy(base_array) * multiplier
        result[mask_array == 0] = base_nodata
        return result

    if multiplier is None:
        multiplier = 1
        target_datatype = base_raster_info['datatype']
    else:
        target_datatype = gdal.GDT_Float32

    pygeoprocessing.raster_calculator(
        [base_raster_path_band, (mask_raster_path, 1), (multiplier, 'raw')],
        mask_op, target_mask_raster_path, target_datatype, base_nodata,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)

    os.remove(mask_raster_path)


def eastern_steppe_future_climate():
    """Run RPM with real future climate and 2018 animal densities."""
    cmip6_model = 'CanESM5'  # 'GFDL-ESM4'
    # outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/{}_ssp370_2061-2080_zerosd".format(cmip6_model)
    outer_dir = "C:/Users/ginge/Desktop/debugging/future_24months"
    workspace_dir = os.path.join(outer_dir, 'RPM_workspace')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    aoi_path = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/southeastern_aimags_aoi_WGS84.shp"

    # create temporary climate directories, export each band to separate raster
    # future_precip_path = os.path.join(
    #     "E:/GIS_local_archive/General_useful_data/Worldclim_future_climate/cmip6/share/spatial03/worldclim/cmip6/7_fut/2.5m",
    #     cmip6_model,
    #     "ssp370/wc2.1_2.5m_prec_{}_ssp370_2061-2080.tif".format(cmip6_model))
    # future_tmin_path = os.path.join(
    #     "E:/GIS_local_archive/General_useful_data/Worldclim_future_climate/cmip6/share/spatial03/worldclim/cmip6/7_fut/2.5m",
    #     cmip6_model,
    #     "ssp370/wc2.1_2.5m_tmin_{}_ssp370_2061-2080.tif".format(cmip6_model))
    # future_tmax_path = os.path.join(
    #     "E:/GIS_local_archive/General_useful_data/Worldclim_future_climate/cmip6/share/spatial03/worldclim/cmip6/7_fut/2.5m",
    #     cmip6_model,
    #     "ssp370/wc2.1_2.5m_tmax_{}_ssp370_2061-2080.tif".format(cmip6_model))
    # precip_dir = tempfile.mkdtemp(dir=outer_dir)
    # tmin_dir = tempfile.mkdtemp(dir=outer_dir)
    # tmax_dir = tempfile.mkdtemp(dir=outer_dir)
    # for (source_path, out_dir, label, multiplier) in [
    #         (future_precip_path, precip_dir, 'precip', 0.1),
    #         (future_tmin_path, tmin_dir, 'tmin', None),
    #         (future_tmax_path, tmax_dir, 'tmax', None)]:
    #     for band in range(1, 13):
    #         target_raster_path = os.path.join(
    #             out_dir, '{}_2016_{}.tif'.format(label, band))
    #         mask_and_divide(
    #             (source_path, band), aoi_path, target_raster_path,
    #             multiplier, all_touched=True)

    precip_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/CanESM5_ssp370_2061-2080/tmpjyovbsmf"
    tmin_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/CanESM5_ssp370_2061-2080/tmpazi_z8tq"
    tmax_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/CanESM5_ssp370_2061-2080/tmpx4z13wzi"

    ahlborn_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs"
    eastern_steppe_dir = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop"
    rpm_args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 24,
        'aoi_path': aoi_path,
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            eastern_steppe_dir, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            eastern_steppe_dir, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            eastern_steppe_dir, 'soil', 'pH.tif'),
        'clay_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'sand.tif'),
        'precip_dir': precip_dir,
        'min_temp_dir': tmin_dir,
        'max_temp_dir': tmax_dir,
        'monthly_vi_path_pattern': (
            "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_Ahlborn_sites/exported_to_tif/ndvi_<year>_<month>.tif"),
        'site_param_table': os.path.join(
            ahlborn_dir, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            eastern_steppe_dir, 'site_index.tif'),
        'veg_trait_path': os.path.join(ahlborn_dir, 'pft_trait.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            eastern_steppe_dir, "pft<PFT>.tif"),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/sfu.csv",
        'animal_grazing_areas_path': os.path.join(
            eastern_steppe_dir, "sfu_per_soum.shp"),
        'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/initial_conditions",
        # 'animal_density': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/sfu_per_ha_NSO_2018.tif",
        'animal_density': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/zero_sfu_per_ha.tif",
        'crude_protein': CRUDE_PROTEIN,
    }
    forage.execute(rpm_args)

    # clean up
    # shutil.rmtree(precip_dir)
    # shutil.rmtree(tmin_dir)
    # shutil.rmtree(tmax_dir)


def eastern_steppe_current_climate():
    """Run eastern steppe aoi with current worldclim."""
    # outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/Eastern_steppe_scenarios/current_zerosd"
    outer_dir = "C:/Users/ginge/Desktop/debugging/current"
    aoi_path = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop/southeastern_aimags_aoi_WGS84.shp"
    # aoi_path = "C:/Users/ginge/Desktop/debugging/small_aoi.shp"

    ahlborn_dir = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/Ahlborn_sites/RPM_inputs"
    eastern_steppe_dir = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop"
    rpm_args = {
        'workspace_dir': os.path.join(outer_dir, 'RPM_workspace'),
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 12,
        'aoi_path': aoi_path,
        'management_threshold': 100,
        'proportion_legume_path': os.path.join(
            eastern_steppe_dir, 'prop_legume.tif'),
        'bulk_density_path': os.path.join(
            eastern_steppe_dir, 'soil', 'bulkd.tif'),
        'ph_path': os.path.join(
            eastern_steppe_dir, 'soil', 'pH.tif'),
        'clay_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'clay.tif'),
        'silt_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'silt.tif'),
        'sand_proportion_path': os.path.join(
            eastern_steppe_dir, 'soil', 'sand.tif'),
        'precip_dir': "E:/GIS_local/Mongolia/Worldclim/Worldclim_baseline",
        'min_temp_dir': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_min",
        'max_temp_dir': "E:/GIS_local_archive/General_useful_data/Worldclim_2.0/temperature_max",
        'monthly_vi_path_pattern': (
            "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_Ahlborn_sites/exported_to_tif/ndvi_<year>_<month>.tif"),
        'site_param_table': os.path.join(
            ahlborn_dir, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            eastern_steppe_dir, 'site_index.tif'),
        'veg_trait_path': os.path.join(ahlborn_dir, 'pft_trait.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            eastern_steppe_dir, "pft<PFT>.tif"),
        'animal_trait_path': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/sfu.csv",
        'animal_grazing_areas_path': os.path.join(
            eastern_steppe_dir, "sfu_per_soum.shp"),
        # 'animal_grazing_areas_path': aoi_path,
        'initial_conditions_dir': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/initial_conditions",
        # 'animal_density': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/sfu_per_ha_NSO_2018.tif",
        'animal_density': "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/eastern_steppe_regular_grid/RPM_inputs/zero_sfu_per_ha.tif",
        'crude_protein': CRUDE_PROTEIN,
    }
    forage.execute(rpm_args)


def main():
    """Program entry point."""
    # wcs_monitoring_sites_workflow()
    # ahlborn_sites_workflow()
    # ahlborn_sites_collect_precip()
    # gobi_scenarios()
    Ahlborn_scenarios()
    # Laikipia_scenarios()
    # Laikipia_NDVI_test()
    # launch_RPM_OPC()
    # Ahlborn_zerosd_biomass_vs_Century()
    # eastern_steppe_future_climate()
    # eastern_steppe_current_climate()


if __name__ == "__main__":
    __spec__ = None  # for running with pdb
    # extend()
    main()
