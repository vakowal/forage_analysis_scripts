"""Profile InVEST forage model."""

import tempfile
import os

from natcap.invest import forage

SAMPLE_DATA = "C:/Users/ginge/Dropbox/sample_inputs"
REGRESSION_DATA = "C:/Users/ginge/Documents/NatCap/regression_test_data"


def generate_base_args(workspace_dir):
    """Generate a base sample args dict for forage model."""
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 1,
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
        'site_param_table': os.path.join(
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
        'initial_conditions_dir': os.path.join(
            SAMPLE_DATA, 'initialization_data'),
    }
    return args


workspace_dir = tempfile.mkdtemp()
args = generate_base_args(workspace_dir)
forage.execute(args)
