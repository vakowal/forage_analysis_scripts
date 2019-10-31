"""Generate example outputs from Rangeland model."""
import os
import tempfile

import numpy
from osgeo import gdal

import pygeoprocessing

PRECIP_DIR = "C:/Users/ginge/Dropbox/sample_inputs/CHIRPS_div_by_10"
_IC_NODATA = float(numpy.finfo('float32').min)

numpy.random.seed(100)


def zero_to_nodata(base_raster):
    """Reclassify zero values in `base_raster` to nodata.

    Parameters:
        base_raster (numpy.ndarray): raster containing zero values

    Returns:
        target_raster, a copy of base_raster with zero values replaced by
            NoData

    """
    target_raster = base_raster.copy()
    target_raster[target_raster == 0] = _IC_NODATA
    return target_raster


def precip_to_correlated_output(precip, multiplier, jitter):
    """Derive a random raster from precipitation.

    Derive a raster of random values that are correlated with precipitation.
    Do this by multiplying the precipitation raster by a constant multiplier
    and adding randomly generated constants inside the range given by
    (-jitter, jitter).

    Parameters:
        precip (numpy.ndarray): precipitation raster
        multiplier (float): constant by which to multiply the precipitation
            raster
        jitter (float): amount by which to jitter output values.

    Returns:
        correlated_output, a raster of random values that are correlated with
            the precipitation raster

    """
    valid_mask = (precip != _IC_NODATA)
    random_array = numpy.random.uniform(-jitter, jitter, precip.shape)

    correlated_output = numpy.empty(precip.shape, dtype=numpy.float32)
    correlated_output[:] = _IC_NODATA
    correlated_output[valid_mask] = (
        precip[valid_mask] * multiplier + random_array[valid_mask])
    return correlated_output


def precip_to_animal_density(precip, average_value, jitter):
    """Derive a random raster of animal densities.

    Derive a raster of values randomly jittered around an average value.

    Parameters:
        precip (numpy.ndarray): precipitation
        average_value (float): average value of the output
        jitter (float): amount by which to jitter output values around the
            average values

    Returns:
        animal_density, a raster of values randomly jittered around the average
            value

    """
    valid_mask = (precip != _IC_NODATA)
    random_array = numpy.random.uniform(-jitter, jitter, precip.shape)

    animal_density = numpy.empty(precip.shape, dtype=numpy.float32)
    animal_density[:] = _IC_NODATA
    animal_density[valid_mask] = average_value + random_array[valid_mask]
    return animal_density


def workflow(save_dir):
    """Create example outputs from a series of rasters covering study area.

    Use precipitation rasters as templates to create example outputs from the
    Rangeland model.  Create model outputs representing, for each modeled time
        step:
        - total potential biomass in the absence of grazing
        - total standing biomass after offtake by grazing animals
        - animal density
        - diet sufficiency of grazing animals

    Parameters:
        save_dir (string): path to directory where outputs should be saved. If
            the directory exists, it will be overwritten

    Returns:
        None

    """
    year = 2016
    month_series = range(1, 13)
    total_potential_biomass_multiplier = 48.8
    total_standing_biomass_multiplier = 45.25
    biomass_jitter = 3.
    diet_sufficiency_multiplier = 0.28
    diet_sufficiency_jitter = 0.01
    avg_animal_density = 0.0175
    animal_density_jitter = 0.005

    # twelve months of precipitation rasters covering the study area
    precip_basename_list = [
        'chirps-v2.0.{}.{:02d}.tif'.format(year, month) for month in
        month_series]

    # reclassify 0 to NoData in CHIRPS rasters
    output_precip_dir = os.path.join(save_dir, 'precip')
    if not os.path.exists(output_precip_dir):
        os.makedirs(output_precip_dir)
    for bn in precip_basename_list:
        base_raster = os.path.join(PRECIP_DIR, bn)
        target_raster = os.path.join(output_precip_dir, bn)
        pygeoprocessing.raster_calculator(
            [(base_raster, 1)], zero_to_nodata, target_raster,
            gdal.GDT_Float32, _IC_NODATA)

    # generate outputs
    for month in month_series:
        precip_raster = os.path.join(
            output_precip_dir, 'chirps-v2.0.{}.{:02d}.tif'.format(year, month))

        total_potential_biomass_path = os.path.join(
            save_dir, 'potential_biomass_{}_{:02d}.tif'.format(year, month))
        pygeoprocessing.raster_calculator(
            [(precip_raster, 1)] + [(path, 'raw') for path in [
                total_potential_biomass_multiplier,
                biomass_jitter]],
            precip_to_correlated_output, total_potential_biomass_path,
            gdal.GDT_Float32, _IC_NODATA)

        total_standing_biomass_path = os.path.join(
            save_dir, 'standing_biomass_{}_{:02d}.tif'.format(year, month))
        pygeoprocessing.raster_calculator(
            [(precip_raster, 1)] + [(path, 'raw') for path in [
                total_standing_biomass_multiplier,
                biomass_jitter]],
            precip_to_correlated_output, total_standing_biomass_path,
            gdal.GDT_Float32, _IC_NODATA)

        diet_sufficiency_path = os.path.join(
            save_dir, 'diet_sufficiency_{}_{:02d}.tif'.format(year, month))
        pygeoprocessing.raster_calculator(
            [(precip_raster, 1)] + [(path, 'raw') for path in [
                diet_sufficiency_multiplier,
                diet_sufficiency_jitter]],
            precip_to_correlated_output, diet_sufficiency_path,
            gdal.GDT_Float32, _IC_NODATA)

        animal_density_path = os.path.join(
            save_dir, 'animal_density_{}_{:02d}.tif'.format(year, month))
        pygeoprocessing.raster_calculator(
            [(precip_raster, 1)] + [(path, 'raw') for path in [
                avg_animal_density,
                animal_density_jitter]],
            precip_to_animal_density, animal_density_path,
            gdal.GDT_Float32, _IC_NODATA)


if __name__ == "__main__":
    save_dir = "C:/Users/ginge/Dropbox/sample_inputs/output"
    workflow(save_dir)
