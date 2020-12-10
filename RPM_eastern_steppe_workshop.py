"""RPM run for WCS Eastern Steppe workshop."""
import os

from rangeland_production import forage

# inputs
# DATA_DIR = "E:/GIS_local/Mongolia/WCS_Eastern_Steppe_workshop"
DATA_DIR = "C:/Users/ginge/Desktop/training/RPM_input"

# input directory for Gobi runs
GOBI_DATA_DIR = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/model_inputs/RPM_initialized_from_Century"


def generate_base_args(workspace_dir):
    """Generate an args dict of basic inputs for rangeland production model."""
    args = {
        'workspace_dir': workspace_dir,
        'results_suffix': "",
        'starting_month': 1,
        'starting_year': 2016,
        'n_months': 12,
        'aoi_path': os.path.join(
            DATA_DIR, 'southeastern_aimags_aoi_WGS84.shp'),
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
            DATA_DIR, 'precipitation', 'worldclim',
            'twc2.0_30s_prec_<year>-<month>.tif'),
        'min_temp_path_pattern': os.path.join(
            DATA_DIR, 'temperature', 'twc2.0_30s_tmin_<month>.tif'),
        'max_temp_path_pattern': os.path.join(
            DATA_DIR, 'temperature', 'twc2.0_30s_tmax_<month>.tif'),
        'monthly_vi_path_pattern': os.path.join(
            DATA_DIR, 'NDVI', 'NDVI_<year>_<month>.tif'),
        'site_param_table': os.path.join(
            DATA_DIR, 'site_parameters.csv'),
        'site_param_spatial_index_path': os.path.join(
            DATA_DIR, 'site_index.tif'),
        'veg_trait_path': os.path.join(
            DATA_DIR, 'pft_parameters.csv'),
        'veg_spatial_composition_path_pattern': os.path.join(
            DATA_DIR, 'pft<PFT>.tif'),
        'animal_trait_path': os.path.join(
            DATA_DIR, 'animal_parameters.csv'),
        'animal_grazing_areas_path': os.path.join(
            DATA_DIR, 'sfu_per_soum.shp'),
        'pft_initial_table': os.path.join(DATA_DIR, 'pft_initial_table.csv'),
        'site_initial_table': os.path.join(DATA_DIR, 'site_initial_table.csv'),
    }
    return args


def main():
    """Launch model and check results."""
    outer_dir = "C:/Users/ginge/Documents/NatCap/GIS_local/Mongolia/RPM_eastern_steppe_workshop"
    workspace_dir = os.path.join(outer_dir, 'baseline')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    rpm_args = generate_base_args(workspace_dir)
    forage.execute(rpm_args)


if __name__ == "__main__":
    __spec__ = None  # for running with pdb
    main()
