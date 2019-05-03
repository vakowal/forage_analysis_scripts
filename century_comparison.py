

def test_one_step_no_grazing(self):
    """Test change in state variables after one step against Century.

    Run the model for one timestep without grazing. Compare state variables
    at the end of one time step to state variables generated from point-
    based Century model.

    Raises:
        ???

    Returns:
        None
    """
    from natcap.invest import forage

    args = foragetests.generate_base_args(self.workspace_dir)
    if not os.path.exists(SAMPLE_DATA):
        self.fail(
            "Sample input directory not found at %s" % SAMPLE_DATA)
    if not os.path.exists(REGRESSION_DATA):
        self.fail(
            "Testing data not found at %s" % REGRESSION_DATA)

    forage.execute(args)

    ending_sv_dir = os.path.join(
        args['workspace_dir'], 'state_variables_m0')
    sv_path_list = [
        f for f in os.listdir(ending_sv_dir) if f.endswith('.tif')]
    summary_dict = {
        'state_variable': [],
        'min': [],
        'max': [],
        'count': [],
        'nodata_count': [],
    }
    for sv_path in sv_path_list:
        zonal_dict = calc_raster_difference_stats(
            os.path.join(ending_sv_dir, sv_path),
            os.path.join(REGRESSION_DATA, sv_path),
            args['aoi_path'])
        summary_dict['state_variable'].append(sv_path)
        summary_dict['min'].append(zonal_dict[0]['min'])
        summary_dict['max'].append(zonal_dict[0]['max'])
        summary_dict['count'].append(zonal_dict[0]['count'])
        summary_dict['nodata_count'].append(zonal_dict[0]['nodata_count'])
    diff_df = pandas.DataFrame(summary_dict).set_index('state_variable')
    diff_df.to_csv("C:/Users/ginge/Desktop/raster_diff_m0.csv")

def test_decomposition(self):
    """Test `_decomposition`.

    Test the function `_decomposition` to calculate the change in state
    variables following one decomposition step.  Compare modified state
    variables to values calculated by point version of decomposition.

    Raises:
        AssertionError if change in state variable calculated by
            `_decomposition` does not match change calculated by point-
            based version

    Returns:
        None
    """
    def generate_model_inputs_from_point_inputs(
            inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
            pp_reg_vals, rnew_dict):
        """Generate model inputs for `_decomposition` from test inputs."""
        nrows = 1
        ncols = 1

        aligned_inputs = {
            'precip_{}'.format(month_index): os.path.join(
                self.workspace_dir, 'precip.tif'),
            'min_temp_{}'.format(current_month): os.path.join(
                self.workspace_dir, 'min_temp.tif'),
            'max_temp_{}'.format(current_month): os.path.join(
                self.workspace_dir, 'max_temp.tif'),
            'ph_path': os.path.join(self.workspace_dir, 'pH.tif'),
            'site_index': os.path.join(
                self.workspace_dir, 'site_index.tif'),
        }
        create_random_raster(
            aligned_inputs['precip_{}'.format(month_index)],
            inputs['precip'], inputs['precip'], nrows=nrows, ncols=ncols)
        create_random_raster(
            aligned_inputs['min_temp_{}'.format(current_month)],
            inputs['min_temp'], inputs['min_temp'],
            nrows=nrows, ncols=ncols)
        create_random_raster(
            aligned_inputs['max_temp_{}'.format(current_month)],
            inputs['max_temp'], inputs['max_temp'], nrows=nrows,
            ncols=ncols)
        create_random_raster(
            aligned_inputs['ph_path'], inputs['pH'], inputs['pH'],
            nrows=nrows, ncols=ncols)
        create_random_raster(
            aligned_inputs['site_index'], 1, 1, nrows=nrows, ncols=ncols)

        site_param_table = {1: {}}
        for key, value in params.iteritems():
            site_param_table[1][key] = value

        year_reg = {
            'annual_precip_path': os.path.join(
                self.workspace_dir, 'annual_precip.tif'),
            'baseNdep_path': os.path.join(
                self.workspace_dir, 'baseNdep.tif'),
        }
        create_random_raster(
            year_reg['annual_precip_path'], year_reg_vals['annual_precip'],
            year_reg_vals['annual_precip'], nrows=nrows, ncols=ncols)
        create_random_raster(
            year_reg['baseNdep_path'], year_reg_vals['baseNdep'],
            year_reg_vals['baseNdep'], nrows=nrows, ncols=ncols)

        month_reg = {
            'snowmelt': os.path.join(self.workspace_dir, 'snowmelt.tif'),
            'amov_2': os.path.join(self.workspace_dir, 'amov_2.tif'),
            'bgwfunc': os.path.join(self.workspace_dir, 'bgwfunc.tif'),
        }
        create_random_raster(
            month_reg['snowmelt'], month_reg_vals['snowmelt'],
            month_reg_vals['snowmelt'], nrows=nrows, ncols=ncols)
        create_random_raster(
            month_reg['amov_2'], month_reg_vals['amov_2'],
            month_reg_vals['amov_2'], nrows=nrows, ncols=ncols)

        prev_sv_reg = {}
        sv_reg = {}
        for state_var in [
                'minerl_1_1', 'minerl_1_2', 'snow', 'avh2o_3', 'parent_2',
                'secndy_2', 'occlud']:
            prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}_p.tif'.format(state_var))
            sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}.tif'.format(state_var))
            create_random_raster(
                prev_sv_reg['{}_path'.format(state_var)],
                state_var_dict[state_var], state_var_dict[state_var],
                nrows=nrows, ncols=ncols)
            create_random_raster(
                sv_reg['{}_path'.format(state_var)],
                state_var_dict[state_var], state_var_dict[state_var],
                nrows=nrows, ncols=ncols)
        for compartment in ['strlig']:
            for lyr in [1, 2]:
                state_var = '{}_{}'.format(compartment, lyr)
                prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}_p.tif'.format(state_var))
                sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}.tif'.format(state_var))
                create_random_raster(
                    prev_sv_reg['{}_path'.format(state_var)],
                    state_var_dict[state_var], state_var_dict[state_var],
                    nrows=nrows, ncols=ncols)
        for compartment in ['som3']:
            state_var = '{}c'.format(compartment)
            prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}_p.tif'.format(state_var))
            sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}.tif'.format(state_var))
            create_random_raster(
                prev_sv_reg['{}_path'.format(state_var)],
                state_var_dict[state_var],
                state_var_dict[state_var],
                nrows=nrows, ncols=ncols)
            for iel in [1, 2]:
                state_var = '{}e_{}'.format(compartment, iel)
                prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}_p.tif'.format(state_var))
                sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}.tif'.format(state_var))
                create_random_raster(
                    prev_sv_reg['{}_path'.format(state_var)],
                    state_var_dict[state_var],
                    state_var_dict[state_var],
                    nrows=nrows, ncols=ncols)
        for compartment in ['struc', 'metab', 'som1', 'som2']:
            for lyr in [1, 2]:
                state_var = '{}c_{}'.format(compartment, lyr)
                prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}_p.tif'.format(state_var))
                sv_reg['{}_path'.format(state_var)] = os.path.join(
                    self.workspace_dir, '{}.tif'.format(state_var))
                create_random_raster(
                    prev_sv_reg['{}_path'.format(state_var)],
                    state_var_dict[state_var],
                    state_var_dict[state_var],
                    nrows=nrows, ncols=ncols)
                for iel in [1, 2]:
                    state_var = '{}e_{}_{}'.format(compartment, lyr, iel)
                    prev_sv_reg['{}_path'.format(state_var)] = (
                        os.path.join(
                            self.workspace_dir,
                            '{}_p.tif'.format(state_var)))
                    sv_reg['{}_path'.format(state_var)] = os.path.join(
                        self.workspace_dir, '{}.tif'.format(state_var))
                    create_random_raster(
                        prev_sv_reg['{}_path'.format(state_var)],
                        state_var_dict[state_var],
                        state_var_dict[state_var],
                        nrows=nrows, ncols=ncols)
        for lyr in xrange(1, params['nlayer'] + 1):
            state_var = 'minerl_{}_2'.format(lyr)
            prev_sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}_p.tif'.format(state_var))
            sv_reg['{}_path'.format(state_var)] = os.path.join(
                self.workspace_dir, '{}.tif'.format(state_var))
            create_random_raster(
                prev_sv_reg['{}_path'.format(state_var)],
                state_var_dict[state_var],
                state_var_dict[state_var],
                nrows=nrows, ncols=ncols)
        pp_reg = {
            'rnewas_1_1_path': os.path.join(
                self.workspace_dir, 'rnewas_1_1.tif'),
            'rnewas_1_2_path': os.path.join(
                self.workspace_dir, 'rnewas_1_2.tif'),
            'rnewas_2_1_path': os.path.join(
                self.workspace_dir, 'rnewas_2_1.tif'),
            'rnewas_2_2_path': os.path.join(
                self.workspace_dir, 'rnewas_2_2.tif'),
            'rnewbs_1_1_path': os.path.join(
                self.workspace_dir, 'rnewbs_1_1.tif'),
            'rnewbs_1_2_path': os.path.join(
                self.workspace_dir, 'rnewbs_1_2.tif'),
            'rnewbs_2_1_path': os.path.join(
                self.workspace_dir, 'rnewbs_2_1.tif'),
            'rnewbs_2_2_path': os.path.join(
                self.workspace_dir, 'rnewbs_2_2.tif'),
        }
        for key in rnew_dict.iterkeys():
            create_random_raster(
                pp_reg['{}_path'.format(key)], rnew_dict[key],
                rnew_dict[key], nrows=nrows, ncols=ncols)
        pp_reg['eftext_path'] = os.path.join(
            self.workspace_dir, 'eftext.tif')
        pp_reg['orglch_path'] = os.path.join(
            self.workspace_dir, 'orglch.tif')
        pp_reg['fps1s3_path'] = os.path.join(
            self.workspace_dir, 'fps1s3.tif')
        pp_reg['fps2s3_path'] = os.path.join(
            self.workspace_dir, 'fps2s3.tif')
        pp_reg['p1co2_2_path'] = os.path.join(
            self.workspace_dir, 'p1co2_2.tif')
        create_random_raster(
            pp_reg['eftext_path'], pp_reg_vals['eftext'],
            pp_reg_vals['eftext'], nrows=nrows, ncols=ncols)
        create_random_raster(
            pp_reg['orglch_path'], pp_reg_vals['orglch'],
            pp_reg_vals['orglch'], nrows=nrows, ncols=ncols)
        create_random_raster(
            pp_reg['fps1s3_path'], pp_reg_vals['fps1s3'],
            pp_reg_vals['fps1s3'], nrows=nrows, ncols=ncols)
        create_random_raster(
            pp_reg['fps2s3_path'], pp_reg_vals['fps2s3'],
            pp_reg_vals['fps2s3'], nrows=nrows, ncols=ncols)
        create_random_raster(
            pp_reg['p1co2_2_path'], pp_reg_vals['p1co2_2'],
            pp_reg_vals['p1co2_2'], nrows=nrows, ncols=ncols)

        input_dict = {
            'aligned_inputs': aligned_inputs,
            'site_param_table': site_param_table,
            'year_reg': year_reg,
            'month_reg': month_reg,
            'prev_sv_reg': prev_sv_reg,
            'sv_reg': sv_reg,
            'pp_reg': pp_reg,
        }
        return input_dict
    from natcap.invest import forage

    # known inputs
    current_month = 4
    month_index = 2
    inputs = {
        'min_temp': 3.2,
        'max_temp': 17.73,
        'precip': 30.5,
        'pH': 6.84,
    }
    params = {
        'fwloss_4': 0.8,
        'teff_1': 15.4,
        'teff_2': 11.75,
        'teff_3': 29.7,
        'teff_4': 0.031,
        'drain': 1,
        'aneref_1': 1.5,
        'aneref_2': 3,
        'aneref_3': 0.3,
        'epnfs_2': 30.,
        'sorpmx': 2,
        'pslsrb': 1,
        'strmax_1': 5000,
        'dec1_1': 3.9,
        'pligst_1': 3,
        'rsplig': 0.3,
        'ps1co2_1': 0.45,
        'strmax_2': 5000,
        'dec1_2': 4.9,
        'pligst_2': 3,
        'ps1co2_2': 0.55,
        'pcemic1_1_1': 16,
        'pcemic1_2_1': 10,
        'pcemic1_3_1': 0.02,
        'pcemic1_1_2': 200,
        'pcemic1_2_2': 99,
        'pcemic1_3_2': 0.0015,
        'dec2_1': 14.8,
        'pmco2_1': 0.55,
        'varat1_1_1': 14,
        'varat1_2_1': 3,
        'varat1_3_1': 2,
        'varat1_1_2': 150,
        'varat1_2_2': 30,
        'varat1_3_2': 2,
        'dec2_2': 18.5,
        'pmco2_2': 0.55,
        'rad1p_1_1': 12,
        'rad1p_2_1': 3,
        'rad1p_3_1': 5,
        'rad1p_1_2': 220,
        'rad1p_2_2': 5,
        'rad1p_3_2': 100,
        'dec3_1': 6,
        'p1co2a_1': 0.6,
        'varat22_1_1': 20,
        'varat22_2_1': 12,
        'varat22_3_1': 2,
        'varat22_1_2': 400,
        'varat22_2_2': 100,
        'varat22_3_2': 2,
        'dec3_2': 7.3,
        'animpt': 5,
        'varat3_1_1': 8,
        'varat3_2_1': 6,
        'varat3_3_1': 2,
        'varat3_1_2': 200,
        'varat3_2_2': 50,
        'varat3_3_2': 2,
        'omlech_3': 60,
        'dec5_2': 0.2,
        'p2co2_2': 0.55,
        'dec5_1': 0.2,
        'p2co2_1': 0.55,
        'dec4': 0.0045,
        'p3co2': 0.55,
        'cmix': 0.5,
        'pparmn_2': 0.0001,
        'psecmn_2': 0.0022,
        'nlayer': 5,
        'pmnsec_2': 0,
        'psecoc1': 0,
        'psecoc2': 0,
        'vlossg': 1,
    }
    state_var_dict = {
        'snow': 0.,
        'avh2o_3': 3.110,
        'strlig_1': 0.3779,
        'strlig_2': 0.2871,
        'minerl_1_1': 6.4143,
        'minerl_1_2': 33.1954,
        'strucc_1': 156.0546,
        'struce_1_1': 0.7803,
        'struce_1_2': 0.3121,
        'som2c_1': 92.9935,
        'som2e_1_1': 4.2466,
        'som2e_1_2': 0.1328,
        'som1c_1': 12.8192,
        'som1e_1_1': 1.5752,
        'som1e_1_2': 0.1328,
        'strucc_2': 163.2008,
        'struce_2_1': 0.816,
        'struce_2_2': 0.3264,
        'som2c_2': 922.1382,
        'som2e_2_1': 60.0671,
        'som2e_2_2': 5.6575,
        'som1c_2': 39.1055,
        'som1e_2_1': 12.2767,
        'som1e_2_2': 1.2263,
        'metabc_1': 13.7579,
        'metabe_1_1': 1.1972,
        'metabe_1_2': 0.0351,
        'metabc_2': 9.8169,
        'metabe_2_1': 0.6309,
        'metabe_2_2': 0.088,
        'som3c': 544.8848,
        'som3e_1': 87.4508,
        'som3e_2': 79.917,
        'parent_2': 100.,
        'secndy_2': 50.,
        'occlud': 50.,
    }
    for lyr in xrange(1, params['nlayer'] + 1):
        state_var_dict['minerl_{}_2'.format(lyr)] = 20.29

    year_reg_vals = {
        'annual_precip': 230.,
        'baseNdep': 24.,
    }
    month_reg_vals = {
        'snowmelt': 0.29,
        'amov_2': 0.481,
    }
    pp_reg_vals = {
        'eftext': 0.15,
        'orglch': 0.01,
        'fps1s3': 0.58,
        'fps2s3': 0.58,
        'p1co2_2': 0.55,
    }
    rnew_dict = {
        'rnewas_1_1': 210.8,
        'rnewas_2_1': 540.2,
        'rnewas_1_2': 190.3,
        'rnewas_2_2': 520.8,
        'rnewbs_1_1': 210.8,
        'rnewbs_2_1': 540.2,
        'rnewbs_1_2': 190.3,
        'rnewbs_2_2': 520.8,
    }
    pevap = 9.324202
    input_dict = generate_model_inputs_from_point_inputs(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict)

    sv_mod_point = decomposition_point(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict, pevap)

    sv_mod_raster = forage._decomposition(
        input_dict['aligned_inputs'], current_month, month_index,
        input_dict['site_param_table'], input_dict['year_reg'],
        input_dict['month_reg'], input_dict['prev_sv_reg'],
        input_dict['sv_reg'], input_dict['pp_reg'])

    for compartment in ['struc', 'metab', 'som1', 'som2']:
        tolerance = 0.0001
        for lyr in [1, 2]:
            state_var = '{}c_{}'.format(compartment, lyr)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
            for iel in [1, 2]:
                state_var = '{}e_{}_{}'.format(compartment, lyr, iel)
                point_value = sv_mod_point[state_var]
                model_res_path = sv_mod_raster['{}_path'.format(state_var)]
                self.assert_all_values_in_raster_within_range(
                    model_res_path, point_value - tolerance,
                    point_value + tolerance, _SV_NODATA)
    for compartment in ['som3']:
        state_var = '{}c'.format(compartment)
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)
        for iel in [1, 2]:
            state_var = '{}e_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for compartment in ['minerl_1']:
        tolerance = 0.001
        for iel in [1, 2]:
            state_var = '{}_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for state_var in ['parent_2', 'secndy_2', 'occlud']:
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)

    # no decomposition, mineral ratios are insufficient
    rnew_dict = {
        'rnewas_1_1': 190.,
        'rnewas_2_1': 300.,
        'rnewas_1_2': 140.,
        'rnewas_2_2': 200.,
        'rnewbs_1_1': 190.,
        'rnewbs_2_1': 300.,
        'rnewbs_1_2': 140.,
        'rnewbs_2_2': 200.
    }
    state_var_dict['minerl_1_1'] = 0.00000001
    state_var_dict['minerl_1_2'] = 0.00000001

    input_dict = generate_model_inputs_from_point_inputs(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict)

    sv_mod_point = decomposition_point(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict, pevap)

    sv_mod_raster = forage._decomposition(
        input_dict['aligned_inputs'], current_month, month_index,
        input_dict['site_param_table'], input_dict['year_reg'],
        input_dict['month_reg'], input_dict['prev_sv_reg'],
        input_dict['sv_reg'], input_dict['pp_reg'])

    for compartment in ['struc', 'metab', 'som1', 'som2']:
        tolerance = 0.0001
        for lyr in [1, 2]:
            state_var = '{}c_{}'.format(compartment, lyr)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
            for iel in [1, 2]:
                state_var = '{}e_{}_{}'.format(compartment, lyr, iel)
                point_value = sv_mod_point[state_var]
                model_res_path = sv_mod_raster['{}_path'.format(state_var)]
                self.assert_all_values_in_raster_within_range(
                    model_res_path, point_value - tolerance,
                    point_value + tolerance, _SV_NODATA)
    for compartment in ['som3']:
        state_var = '{}c'.format(compartment)
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)
        for iel in [1, 2]:
            state_var = '{}e_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for compartment in ['minerl_1']:
        tolerance = 0.001
        for iel in [1, 2]:
            state_var = '{}_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for state_var in ['parent_2', 'secndy_2', 'occlud']:
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)

    # decomposition occurs, subsidized by mineral N and P
    rnew_dict = {
        'rnewas_1_1': 210.8,
        'rnewas_2_1': 540.2,
        'rnewas_1_2': 190.3,
        'rnewas_2_2': 520.8,
        'rnewbs_1_1': 210.8,
        'rnewbs_2_1': 540.2,
        'rnewbs_1_2': 190.3,
        'rnewbs_2_2': 520.8
    }
    state_var_dict['minerl_1_1'] = 6.01
    state_var_dict['minerl_1_2'] = 32.87

    state_var_dict['strlig_1'] = 0.2987
    state_var_dict['strlig_2'] = 0.4992

    input_dict = generate_model_inputs_from_point_inputs(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict)

    sv_mod_point = decomposition_point(
        inputs, params, state_var_dict, year_reg_vals, month_reg_vals,
        pp_reg_vals, rnew_dict, pevap)

    sv_mod_raster = forage._decomposition(
        input_dict['aligned_inputs'], current_month, month_index,
        input_dict['site_param_table'], input_dict['year_reg'],
        input_dict['month_reg'], input_dict['prev_sv_reg'],
        input_dict['sv_reg'], input_dict['pp_reg'])

    for compartment in ['struc', 'metab', 'som1', 'som2']:
        tolerance = 0.0001
        for lyr in [1, 2]:
            state_var = '{}c_{}'.format(compartment, lyr)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
            for iel in [1, 2]:
                state_var = '{}e_{}_{}'.format(compartment, lyr, iel)
                point_value = sv_mod_point[state_var]
                model_res_path = sv_mod_raster['{}_path'.format(state_var)]
                self.assert_all_values_in_raster_within_range(
                    model_res_path, point_value - tolerance,
                    point_value + tolerance, _SV_NODATA)
    for compartment in ['som3']:
        state_var = '{}c'.format(compartment)
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)
        for iel in [1, 2]:
            state_var = '{}e_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for compartment in ['minerl_1']:
        tolerance = 0.001
        for iel in [1, 2]:
            state_var = '{}_{}'.format(compartment, iel)
            point_value = sv_mod_point[state_var]
            model_res_path = sv_mod_raster['{}_path'.format(state_var)]
            self.assert_all_values_in_raster_within_range(
                model_res_path, point_value - tolerance,
                point_value + tolerance, _SV_NODATA)
    for state_var in ['parent_2', 'secndy_2', 'occlud']:
        point_value = sv_mod_point[state_var]
        model_res_path = sv_mod_raster['{}_path'.format(state_var)]
        self.assert_all_values_in_raster_within_range(
            model_res_path, point_value - tolerance,
            point_value + tolerance, _SV_NODATA)


def decomposition_point(
        inputs, params, state_var, year_reg, month_reg, pp_reg, rnew_dict,
        pevap):
    """Point implementation of decomposition.

    Parameters:
        inputs (dict): dictionary of input values, including precipitation,
            temperature, and soil pH
        params (dict): dictionary of parameter values
        state_var (dict): dictionary of state variables prior to
            decomposition
        year_reg (dict): dictionary of values that are updated once per year,
            including annual precipitation and annual N deposition
        month_reg (dict): dictionary of values that are shared between
            submodels, including snowmelt
        pp_reg (dict): dictionary of persistent parameters, calculated upon
            model initialization
        rnew_dict (dict): dictionary of required ratios for aboveground
            decomposition, calculated during initialization
        evap (float): reference evapotranspiration

    Returns
        dictionary of modified state variables following decomposition
    """
    gromin_1 = 0  # gross mineralization of N, used outside decomposition
    rprpet = rprpet_point(
        pevap, month_reg['snowmelt'], state_var['avh2o_3'], inputs['precip'])
    defac = defac_point(
        state_var['snow'], inputs['min_temp'], inputs['max_temp'], rprpet,
        params['teff_1'], params['teff_2'], params['teff_3'], params['teff_4'])
    anerb = calc_anerb_point(
        rprpet, pevap, params['drain'], params['aneref_1'], params['aneref_2'],
        params['aneref_3'])

    # pH effect on decomposition for structural material, line 45
    pheff_struc = numpy.clip(
        (0.5 + (1.1 / numpy.pi) *
            numpy.arctan(numpy.pi * 0.7 * (inputs['pH'] - 4.))), 0, 1)

    # pH effect on decomposition for metabolic material, line 158
    pheff_metab = numpy.clip(
        (0.5 + (1.14 / numpy.pi) *
            numpy.arctan(numpy.pi * 0.7 * (inputs['pH'] - 4.8))), 0, 1)

    # calculate aminrl_1, intermediate surface mineral N that is tracked
    # during decomposition
    aminrl_1 = state_var['minerl_1_1']

    # calculate aminrl_2, intermediate surface mineral P that is tracked
    # during decomposition
    fsol = fsfunc_point(
        state_var['minerl_1_2'], params['pslsrb'], params['sorpmx'])
    aminrl_2 = state_var['minerl_1_2'] * fsol

    # monthly N fixation
    state_var['minerl_1_1'] = monthly_N_fixation_point(
        inputs['precip'], year_reg['annual_precip'], year_reg['baseNdep'],
        params['epnfs_2'], state_var['minerl_1_1'])

    for _ in xrange(1):
        # initialize change (delta, d) in state variables for this decomp step
        d_minerl_1_1 = 0  # change in surface mineral N
        d_minerl_1_2 = 0  # change in surface mineral P

        d_strucc_1 = 0  # change in surface structural C
        d_struce_1_1 = 0  # change in surface structural N
        d_struce_1_2 = 0  # change in surface strctural P
        d_som2c_1 = 0  # change in surface SOM2 C
        d_som2e_1_1 = 0  # change in surface SOM2 N
        d_som2e_1_2 = 0  # change in surface SOM2 P
        d_som1c_1 = 0  # change in surface SOM1 C
        d_som1e_1_1 = 0  # change in surface SOM1 N
        d_som1e_1_2 = 0  # change in surface SOM1 P

        d_strucc_2 = 0  # change in soil structural C
        d_struce_2_1 = 0  # change in soil structural N
        d_struce_2_2 = 0  # change in soil structural P
        d_som2c_2 = 0  # change in soil SOM2 C
        d_som2e_2_1 = 0  # change in soil SOM2 N
        d_som2e_2_2 = 0  # change in soil SOM2 P
        d_som1c_2 = 0  # change in soil SOM1 C
        d_som1e_2_1 = 0  # change in soil SOM1 N
        d_som1e_2_2 = 0  # change in soil SOM1 P

        d_metabc_1 = 0  # change in surface metabolic C
        d_metabe_1_1 = 0  # change in surface metabolic N
        d_metabe_1_2 = 0  # change in surface metabolic P

        d_metabc_2 = 0  # change in soil metabolic C
        d_metabe_2_1 = 0  # change in soil metabolic N
        d_metabe_2_2 = 0  # change in soil metabolic P

        d_som3c = 0  # change in passive organic C
        d_som3e_1 = 0  # change in passive organic N
        d_som3e_2 = 0  # change in passive organic P

        d_parent_2 = 0  # change in parent P
        d_secndy_2 = 0  # change in secondary P
        d_occlud = 0  # change in occluded P

        d_minerl_P_dict = {}
        for lyr in xrange(1, params['nlayer'] + 1):
            d_minerl_P_dict['d_minerl_{}_2'.format(lyr)] = 0

        # litdec.f
        # decomposition of strucc_1, surface structural material
        tcflow = (min(
            state_var['strucc_1'], params['strmax_1']) * defac *
            params['dec1_1'] *
            math.exp(-params['pligst_1'] * state_var['strlig_1']) *
            0.020833 * pheff_struc)

        d_strucc_1 += declig_point(
            'd_strucc')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_struce_1_1 += declig_point(
            'd_struce_1')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_struce_1_2 += declig_point(
            'd_struce_2')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_minerl_1_1 += declig_point(
            'd_minerl_1_1')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_minerl_1_2 += declig_point(
            'd_minerl_1_2')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        gromin_1 += declig_point(
            'd_gromin_1')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som2c_1 += declig_point(
            'd_som2c')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som2e_1_1 += declig_point(
            'd_som2e_1')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som2e_1_2 += declig_point(
            'd_som2e_2')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som1c_1 += declig_point(
            'd_som1c')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som1e_1_1 += declig_point(
            'd_som1e_1')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])
        d_som1e_1_2 += declig_point(
            'd_som1e_2')(
                aminrl_1, aminrl_2, state_var['strlig_1'], params['rsplig'],
                params['ps1co2_1'], state_var['strucc_1'],
                tcflow, state_var['struce_1_1'],
                state_var['struce_1_2'], rnew_dict['rnewas_1_1'],
                rnew_dict['rnewas_2_1'], rnew_dict['rnewas_1_2'],
                rnew_dict['rnewas_2_2'], state_var['minerl_1_1'],
                state_var['minerl_1_2'])

        # decomposition of strucc_2, soil structural material: line 99 Litdec.f
        tcflow = (
            min(state_var['strucc_2'], params['strmax_2']) * defac *
            params['dec1_2'] *
            math.exp(-params['pligst_2'] * state_var['strlig_2']) *
            anerb * 0.020833 * pheff_struc)

        d_strucc_2 += declig_point(
            'd_strucc')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_struce_2_1 += declig_point(
            'd_struce_1')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_struce_2_2 += declig_point(
            'd_struce_2')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_minerl_1_1 += declig_point(
            'd_minerl_1_1')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_minerl_1_2 += declig_point(
            'd_minerl_1_2')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        gromin_1 += declig_point(
            'd_gromin_1')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som2c_2 += declig_point(
            'd_som2c')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som2e_2_1 += declig_point(
            'd_som2e_1')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som2e_2_2 += declig_point(
            'd_som2e_2')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som1c_2 += declig_point(
            'd_som1c')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som1e_2_1 += declig_point(
            'd_som1e_1')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])
        d_som1e_2_2 += declig_point(
            'd_som1e_2')(
                aminrl_1, aminrl_2, state_var['strlig_2'], params['rsplig'],
                params['ps1co2_2'], state_var['strucc_2'], tcflow,
                state_var['struce_2_1'], state_var['struce_2_2'],
                rnew_dict['rnewbs_1_1'], rnew_dict['rnewbs_2_1'],
                rnew_dict['rnewbs_1_2'], rnew_dict['rnewbs_2_2'],
                state_var['minerl_1_1'], state_var['minerl_1_2'])

        # decomposition of surface metabolic material: line 136 Litdec.f
        # C/N ratio for surface metabolic residue
        rceto1_1 = agdrat_point(
            state_var['metabe_1_1'], state_var['metabc_1'],
            params['pcemic1_1_1'], params['pcemic1_2_1'],
            params['pcemic1_3_1'])
        # C/P ratio for surface metabolic residue
        rceto1_2 = agdrat_point(
            state_var['metabe_1_2'], state_var['metabc_1'],
            params['pcemic1_1_2'], params['pcemic1_2_2'],
            params['pcemic1_3_2'])
        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['metabc_1'] / state_var['metabe_1_1']) <=
                rceto1_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['metabc_1'] / state_var['metabe_1_2']) <=
                rceto1_2)))  # line 194 Litdec.f
        if decompose_mask:
            tcflow = numpy.clip(
                (state_var['metabc_1'] * defac * params['dec2_1'] * 0.020833 *
                    pheff_metab), 0,
                state_var['metabc_1'])
            co2los = tcflow * params['pmco2_1']
            d_metabc_1 -= tcflow
            # respiration, line 201 Litdec.f
            mnrflo_1 = (
                co2los * state_var['metabe_1_1'] / state_var['metabc_1'])
            d_metabe_1_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = (
                co2los * state_var['metabe_1_2'] / state_var['metabc_1'])
            d_metabe_1_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            net_tosom1 = tcflow - co2los  # line 210 Litdec.f
            d_som1c_1 += net_tosom1
            # N and P flows from metabe_1 to som1e_1, line 222 Litdec.f
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom1, state_var['metabc_1'], rceto1_1,
                    state_var['metabe_1_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom1, state_var['metabc_1'], rceto1_1,
                    state_var['metabe_1_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom1, state_var['metabc_1'], rceto1_1,
                    state_var['metabe_1_1'], state_var['minerl_1_1'])
            # schedule flows
            d_metabe_1_1 -= material_leaving_a
            d_som1e_1_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow

            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom1, state_var['metabc_1'], rceto1_2,
                    state_var['metabe_1_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom1, state_var['metabc_1'], rceto1_2,
                    state_var['metabe_1_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom1, state_var['metabc_1'], rceto1_2,
                    state_var['metabe_1_2'], state_var['minerl_1_2'])
            # schedule flows
            d_metabe_1_2 -= material_leaving_a
            d_som1e_1_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow

        # decomposition of soil metabolic material: line 136 Litdec.f
        # C/N ratio for belowground material to som1
        rceto1_1 = bgdrat_point(
            aminrl_1, params['varat1_1_1'], params['varat1_2_1'],
            params['varat1_3_1'])
        # C/P ratio for soil metabolic material
        rceto1_2 = bgdrat_point(
            aminrl_2, params['varat1_1_2'], params['varat1_2_2'],
            params['varat1_3_2'])
        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['metabc_2'] / state_var['metabe_2_1']) <=
                rceto1_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['metabc_2'] / state_var['metabe_2_2']) <=
                rceto1_2)))  # line 194 Litdec.f
        if decompose_mask:
            tcflow = numpy.clip(
                (state_var['metabc_2'] * defac * params['dec2_2'] * 0.020833 *
                    pheff_metab * anerb),
                0, state_var['metabc_2'])
            co2los = tcflow * params['pmco2_2']
            d_metabc_2 -= tcflow
            # respiration, line 201 Litdec.f
            mnrflo_1 = co2los * state_var['metabe_2_1'] / state_var['metabc_2']
            d_metabe_2_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['metabe_2_2'] / state_var['metabc_2']
            d_metabe_2_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            net_tosom1 = tcflow - co2los  # line 210 Litdec.f
            d_som1c_2 += net_tosom1
            # N and P flows from metabe_2 to som1e_2, line 222 Litdec.f
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom1, state_var['metabc_2'], rceto1_1,
                    state_var['metabe_2_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom1, state_var['metabc_2'], rceto1_1,
                    state_var['metabe_2_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom1, state_var['metabc_2'], rceto1_1,
                    state_var['metabe_2_1'], state_var['minerl_1_1'])
            # schedule flows
            d_metabe_2_1 -= material_leaving_a
            d_som1e_2_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow

            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom1, state_var['metabc_2'], rceto1_2,
                    state_var['metabe_2_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom1, state_var['metabc_2'], rceto1_2,
                    state_var['metabe_2_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom1, state_var['metabc_2'], rceto1_2,
                    state_var['metabe_2_2'], state_var['minerl_1_2'])
            # schedule flows
            d_metabe_2_2 -= material_leaving_a
            d_som1e_2_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow

        # somdec.f
        # surface SOM1 to surface SOM2
        # rceto2_iel: required ratio for flow from surface SOM1 to SOM2
        radds1_1 = (
            params['rad1p_1_1'] + params['rad1p_2_1'] *
            ((state_var['som1c_1'] / state_var['som1e_1_1']) -
                params['pcemic1_2_1']))
        rceto2_1_surface = max(
            (state_var['som1c_1'] / state_var['som1e_1_1'] + radds1_1),
            params['rad1p_3_1'])

        radds1_2 = (
            params['rad1p_1_2'] + params['rad1p_2_2'] *
            ((state_var['som1c_1'] / state_var['som1e_1_2']) -
                params['pcemic1_2_2']))
        rceto2_2_surface = max(
            (state_var['som1c_1'] / state_var['som1e_1_2'] + radds1_2),
            params['rad1p_3_2'])

        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['som1c_1'] / state_var['som1e_1_1']) <=
                rceto2_1_surface)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['som1c_1'] / state_var['som1e_1_2']) <=
                rceto2_2_surface)))  # line 92
        if decompose_mask:
            tcflow = (
                state_var['som1c_1'] * defac * params['dec3_1'] * 0.020833 *
                pheff_struc)
            co2los = tcflow * params['p1co2a_1']
            d_som1c_1 -= tcflow
            # respiration, line 105 Somdec.f
            mnrflo_1 = co2los * state_var['som1e_1_1'] / state_var['som1c_1']
            d_som1e_1_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['som1e_1_2'] / state_var['som1c_1']
            d_som1e_1_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            net_tosom2 = tcflow - co2los
            d_som2c_1 += net_tosom2
            # N and P flows from som1e_1 to som2e_1, line 123 Somdec.f
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom2, state_var['som1c_1'], rceto2_1_surface,
                    state_var['som1e_1_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom2, state_var['som1c_1'], rceto2_1_surface,
                    state_var['som1e_1_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom2, state_var['som1c_1'], rceto2_1_surface,
                    state_var['som1e_1_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som1e_1_1 -= material_leaving_a
            d_som2e_1_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow

            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom2, state_var['som1c_1'], rceto2_2_surface,
                    state_var['som1e_1_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom2, state_var['som1c_1'], rceto2_2_surface,
                    state_var['som1e_1_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom2, state_var['som1c_1'], rceto2_2_surface,
                    state_var['som1e_1_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som1e_1_2 -= material_leaving_a
            d_som2e_1_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow

        # soil SOM1 to soil SOM3 and SOM2
        # required ratios are those pertaining to decomposition to SOM2
        rceto2_1 = bgdrat_point(
            aminrl_1, params['varat22_1_1'], params['varat22_2_1'],
            params['varat22_3_1'])  # line 141 Somdec.f
        rceto2_2 = bgdrat_point(
            aminrl_2, params['varat22_1_2'], params['varat22_2_2'],
            params['varat22_3_2'])

        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['som1c_2'] / state_var['som1e_2_1']) <=
                rceto2_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['som1c_2'] / state_var['som1e_2_2']) <=
                rceto2_2)))  # line 171
        if decompose_mask:
            tcflow = (
                state_var['som1c_2'] * defac * params['dec3_2'] *
                pp_reg['eftext'] * anerb * 0.020833 * pheff_metab)
            co2los = tcflow * pp_reg['p1co2_2']
            d_som1c_2 -= tcflow
            # respiration, line 179 Somdec.f
            mnrflo_1 = co2los * state_var['som1e_2_1'] / state_var['som1c_2']
            d_som1e_2_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['som1e_2_2'] / state_var['som1c_2']
            d_som1e_2_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            tosom3 = (
                tcflow * pp_reg['fps1s3'] *
                (1. + params['animpt'] * (1. - anerb)))
            d_som3c += tosom3
            # C/<iel> ratios of material entering som3e
            rceto3_1 = bgdrat_point(
                aminrl_1, params['varat3_1_1'], params['varat3_2_1'],
                params['varat3_3_1'])
            rceto3_2 = bgdrat_point(
                aminrl_2, params['varat3_1_2'], params['varat3_2_2'],
                params['varat3_3_2'])
            # N and P flows from soil som1e to som3e, line 198
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom3, state_var['som1c_2'], rceto3_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom3, state_var['som1c_2'], rceto3_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom3, state_var['som1c_2'], rceto3_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som1e_2_1 -= material_leaving_a
            d_som3e_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom3, state_var['som1c_2'], rceto3_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom3, state_var['som1c_2'], rceto3_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom3, state_var['som1c_2'], rceto3_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som1e_2_2 -= material_leaving_a
            d_som3e_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow

            # organic leaching: line 204 Somdec.f
            if month_reg['amov_2'] > 0:
                linten = min(
                    (1. - (params['omlech_3'] - month_reg['amov_2']) /
                        params['omlech_3']), 1.)
                cleach = tcflow * pp_reg['orglch'] * linten
                # N leaching: line 230
                rceof1_1 = (state_var['som1c_2'] / state_var['som1e_2_1']) * 2
                orgflow_1 = cleach / rceof1_1
                d_som1e_2_1 -= orgflow_1
                # P leaching: line 232
                rceof1_2 = (state_var['som1c_2'] / state_var['som1e_2_2']) * 35
                orgflow_2 = cleach / rceof1_2
                d_som1e_2_2 -= orgflow_2
            else:
                cleach = 0

            net_tosom2 = tcflow - co2los - tosom3 - cleach
            d_som2c_2 += net_tosom2
            # N and P flows from som1e_2 to som2e_2, line 257
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom2, state_var['som1c_2'], rceto2_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom2, state_var['som1c_2'], rceto2_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom2, state_var['som1c_2'], rceto2_1,
                    state_var['som1e_2_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som1e_2_1 -= material_leaving_a
            d_som2e_2_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    net_tosom2, state_var['som1c_2'], rceto2_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    net_tosom2, state_var['som1c_2'], rceto2_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    net_tosom2, state_var['som1c_2'], rceto2_2,
                    state_var['som1e_2_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som1e_2_2 -= material_leaving_a
            d_som2e_2_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow
        # Soil SOM2 decomposing to soil SOM1 and SOM3, line 269 Somdec.f
        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['som2c_2'] / state_var['som2e_2_1']) <=
                rceto1_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['som2c_2'] / state_var['som2e_2_2']) <=
                rceto1_2)))  # line 298
        if decompose_mask:
            tcflow = (
                state_var['som2c_2'] * defac * params['dec5_2'] * anerb *
                0.020833 * pheff_metab)
            co2los = tcflow * params['p2co2_2']
            d_som2c_2 -= tcflow
            # respiration, line 304 Somdec.f
            mnrflo_1 = co2los * state_var['som2e_2_1'] / state_var['som2c_2']
            d_som2e_2_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['som2e_2_2'] / state_var['som2c_2']
            d_som2e_2_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            tosom3 = (
                tcflow * pp_reg['fps2s3'] *
                (1. + params['animpt'] * (1.0 - anerb)))
            d_som3c += tosom3
            # N and P flows from soil som2e to som3e
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom3, state_var['som2c_2'], rceto3_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom3, state_var['som2c_2'], rceto3_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom3, state_var['som2c_2'], rceto3_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som2e_2_1 -= material_leaving_a
            d_som3e_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom3, state_var['som2c_2'], rceto3_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom3, state_var['som2c_2'], rceto3_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom3, state_var['som2c_2'], rceto3_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som2e_2_2 -= material_leaving_a
            d_som3e_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow

            # rest of the flow from SOM2 goes to SOM1
            tosom1 = tcflow - co2los - tosom3  # line 333 Somdec.f
            d_som1c_2 += tosom1
            # N and P flows from som2e_2 to som1e_2, line 344
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som2c_2'], rceto1_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som2c_2'], rceto1_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som2c_2'], rceto1_1,
                    state_var['som2e_2_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som2e_2_1 -= material_leaving_a
            d_som1e_2_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som2c_2'], rceto1_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som2c_2'], rceto1_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som2c_2'], rceto1_2,
                    state_var['som2e_2_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som2e_2_2 -= material_leaving_a
            d_som1e_2_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow
        # Surface SOM2 decomposes to surface SOM1
        # ratios of material decomposing to SOM1
        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['som2c_1'] / state_var['som2e_1_1']) <=
                rceto1_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['som2c_1'] / state_var['som2e_1_2']) <=
                rceto1_2)))  # line 171
        if decompose_mask:
            tcflow = (
                state_var['som2c_1'] * defac * params['dec5_1'] * 0.020833 *
                pheff_struc)
            co2los = tcflow * params['p2co2_1']  # line 385
            d_som2c_1 -= tcflow
            # respiration, line 388
            mnrflo_1 = co2los * state_var['som2e_1_1'] / state_var['som2c_1']
            d_som2e_1_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['som2e_1_2'] / state_var['som2c_1']
            d_som2e_1_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            tosom1 = tcflow - co2los  # line 393
            d_som1c_1 += tosom1
            # N and P flows from surface som2e to surface som1e, line 404
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som2c_1'], rceto1_1,
                    state_var['som2e_1_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som2c_1'], rceto1_1,
                    state_var['som2e_1_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som2c_1'], rceto1_1,
                    state_var['som2e_1_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som2e_1_1 -= material_leaving_a
            d_som1e_1_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som2c_1'], rceto1_2,
                    state_var['som2e_1_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som2c_1'], rceto1_2,
                    state_var['som2e_1_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som2c_1'], rceto1_2,
                    state_var['som2e_1_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som2e_1_2 -= material_leaving_a
            d_som1e_1_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow
        # SOM3 decomposes to soil SOM1
        pheff_som3 = numpy.clip(
            (0.5 + (1.1 / numpy.pi) *
                numpy.arctan(numpy.pi * 0.7 * (inputs['pH'] - 3.))), 0, 1)
        decompose_mask = (
            ((aminrl_1 > 0.0000001) | (
                (state_var['som3c'] / state_var['som3e_1']) <= rceto1_1)) &
            ((aminrl_2 > 0.0000001) | (
                (state_var['som3c'] / state_var['som3e_2']) <= rceto1_2)))
        if decompose_mask:
            tcflow = (
                state_var['som3c'] * defac * params['dec4'] * anerb *
                0.020833 * pheff_som3)
            co2los = tcflow * params['p3co2'] * anerb  # line 442
            d_som3c -= tcflow
            # respiration, line 446
            mnrflo_1 = co2los * state_var['som3e_1'] / state_var['som3c']
            d_som3e_1 -= mnrflo_1
            d_minerl_1_1 += mnrflo_1
            gromin_1 += mnrflo_1
            mnrflo_2 = co2los * state_var['som3e_2'] / state_var['som3c']
            d_som3e_2 -= mnrflo_2
            d_minerl_1_2 += mnrflo_2

            tosom1 = tcflow - co2los
            d_som1c_2 += tosom1
            # N and P flows from som3e to som1e_2, line 461 Somdec.f
            # N first
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som3c'], rceto1_1,
                    state_var['som3e_1'], state_var['minerl_1_1'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som3c'], rceto1_1,
                    state_var['som3e_1'], state_var['minerl_1_1'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som3c'], rceto1_1,
                    state_var['som3e_1'], state_var['minerl_1_1'])
            # schedule flows
            d_som3e_1 -= material_leaving_a
            d_som1e_2_1 += material_arriving_b
            d_minerl_1_1 += mineral_flow
            if mineral_flow > 0:
                gromin_1 += mineral_flow
            # P second
            material_leaving_a = esched_point(
                'material_leaving_a')(
                    tosom1, state_var['som3c'], rceto1_2,
                    state_var['som3e_2'], state_var['minerl_1_2'])
            material_arriving_b = esched_point(
                'material_arriving_b')(
                    tosom1, state_var['som3c'], rceto1_2,
                    state_var['som3e_2'], state_var['minerl_1_2'])
            mineral_flow = esched_point(
                'mineral_flow')(
                    tosom1, state_var['som3c'], rceto1_2,
                    state_var['som3e_2'], state_var['minerl_1_2'])
            # schedule flows
            d_som3e_2 -= material_leaving_a
            d_som1e_2_2 += material_arriving_b
            d_minerl_1_2 += mineral_flow
        # Surface SOM2 flows to soil SOM2 via mixing
        tcflow = state_var['som2c_1'] * params['cmix'] * defac * 0.020833
        d_som2c_1 -= tcflow
        d_som2c_2 += tcflow
        # N and P flows from som2e_1 to som2e_2, line 495 Somdec.f
        # N first
        mix_ratio_1 = state_var['som2c_1'] / state_var['som2e_1_1']  # line 495
        material_leaving_a = esched_point(
            'material_leaving_a')(
                tcflow, state_var['som2c_1'], mix_ratio_1,
                state_var['som2e_1_1'], state_var['minerl_1_1'])
        material_arriving_b = esched_point(
            'material_arriving_b')(
                tcflow, state_var['som2c_1'], mix_ratio_1,
                state_var['som2e_1_1'], state_var['minerl_1_1'])
        mineral_flow = esched_point(
            'mineral_flow')(
                tcflow, state_var['som2c_1'], mix_ratio_1,
                state_var['som2e_1_1'], state_var['minerl_1_1'])
        # schedule flows
        d_som2e_1_1 -= material_leaving_a
        d_som2e_2_1 += material_arriving_b
        d_minerl_1_1 += mineral_flow
        # P second
        mix_ratio_2 = state_var['som2c_1'] / state_var['som2e_1_2']  # line 495
        material_leaving_a = esched_point(
            'material_leaving_a')(
                tcflow, state_var['som2c_1'], mix_ratio_2,
                state_var['som2e_1_2'], state_var['minerl_1_2'])
        material_arriving_b = esched_point(
            'material_arriving_b')(
                tcflow, state_var['som2c_1'], mix_ratio_2,
                state_var['som2e_1_2'], state_var['minerl_1_2'])
        mineral_flow = esched_point(
            'mineral_flow')(
                tcflow, state_var['som2c_1'], mix_ratio_2,
                state_var['som2e_1_2'], state_var['minerl_1_2'])
        # schedule flows
        d_som2e_1_2 -= material_leaving_a
        d_som2e_2_2 += material_arriving_b
        d_minerl_1_2 += mineral_flow

        # P mineral flows: Pschem.f
        # flow from parent to mineral: line 141
        fparnt = params['pparmn_2'] * state_var['parent_2'] * defac * 0.020833
        d_minerl_1_2 += fparnt
        d_parent_2 -= fparnt

        # flow from secondary to mineral: line 158
        fsecnd = params['psecmn_2'] * state_var['secndy_2'] * defac * 0.020833
        d_minerl_1_2 += fsecnd
        d_secndy_2 -= fsecnd

        # flow from mineral to secondary: line 163
        fmnsec = (
            params['pmnsec_2'] * state_var['minerl_1_2'] * (1 - fsol) * defac *
            0.020833)
        d_minerl_1_2 -= fmnsec
        d_secndy_2 += fmnsec
        for lyr in xrange(2, params['nlayer'] + 1):
            fmnsec = (
                params['pmnsec_2'] * state_var['minerl_{}_2'.format(lyr)] *
                (1 - fsol) * defac * 0.020833)
            d_minerl_P_dict['d_minerl_{}_2'.format(lyr)] -= fmnsec
            d_secndy_2 += fmnsec

        # flow from secondary to occluded: line 171
        fsecoc = params['psecoc1'] * state_var['secndy_2'] * defac * 0.020833
        d_secndy_2 -= fsecoc
        d_occlud += fsecoc

        # flow from occluded to secondary
        focsec = params['psecoc2'] * state_var['occlud'] * defac * 0.020833
        d_occlud -= focsec
        d_secndy_2 += focsec

        # update state variables: perform flows calculated in previous lines
        state_var['minerl_1_1'] += d_minerl_1_1
        state_var['minerl_1_2'] += d_minerl_1_2

        state_var['strucc_1'] += d_strucc_1
        state_var['struce_1_1'] += d_struce_1_1
        state_var['struce_1_2'] += d_struce_1_2
        state_var['som2c_1'] += d_som2c_1
        state_var['som2e_1_1'] += d_som2e_1_1
        state_var['som2e_1_2'] += d_som2e_1_2
        state_var['som1c_1'] += d_som1c_1
        state_var['som1e_1_1'] += d_som1e_1_1
        state_var['som1e_1_2'] += d_som1e_1_2

        state_var['strucc_2'] += d_strucc_2
        state_var['struce_2_1'] += d_struce_2_1
        state_var['struce_2_2'] += d_struce_2_2
        state_var['som2c_2'] += d_som2c_2
        state_var['som2e_2_1'] += d_som2e_2_1
        state_var['som2e_2_2'] += d_som2e_2_2
        state_var['som1c_2'] += d_som1c_2
        state_var['som1e_2_1'] += d_som1e_2_1
        state_var['som1e_2_2'] += d_som1e_2_2

        state_var['metabc_1'] += d_metabc_1
        state_var['metabe_1_1'] += d_metabe_1_1
        state_var['metabe_1_2'] += d_metabe_1_2

        state_var['metabc_2'] += d_metabc_2
        state_var['metabe_2_1'] += d_metabe_2_1
        state_var['metabe_2_2'] += d_metabe_2_2

        state_var['parent_2'] += d_parent_2
        state_var['secndy_2'] += d_secndy_2
        state_var['occlud'] += d_occlud
        for lyr in xrange(2, params['nlayer'] + 1):
            state_var['minerl_{}_2'.format(lyr)] += (
                d_minerl_P_dict['d_minerl_{}_2'.format(lyr)])

        # update aminrl_1
        aminrl_1 = aminrl_1 + state_var['minerl_1_1'] / 2.

        # update aminrl_2
        fsol = fsfunc_point(
            state_var['minerl_1_2'], params['pslsrb'], params['sorpmx'])
        aminrl_2 = aminrl_2 + (state_var['minerl_1_2'] * fsol) / 2.

    # Calculate volatilization loss of nitrogen as a function of
    # gross mineralization: line 323 Simsom.f
    volgm = params['vlossg'] * gromin_1
    state_var['minerl_1_1'] -= volgm
    return state_var