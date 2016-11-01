## guess management by matching biomass in CENTURY
import math
import csv
import os
import sys
import shutil
sys.path.append(
 'C:/Users/Ginger/Documents/Python/invest_forage_dev/src/natcap/invest/forage')
import forage_century_link_utils as cent

def back_calculate_management(site, input_dir, century_dir, out_dir, fix_file,
                              n_months, vary, live_or_total, threshold,
                              max_iterations):
    """Calculate grazing history at a site by adding or removing grazing events
    and modifying grazing intensity until an empirical biomass target is
    reached.  Biomass should be specified in g per square m.  Empirical date
    should be specified as a "CENTURY date": 2012.00 is December of 2011;
    2012.08 is January of 2012, etc. Two decimal points."""
    
    cent.set_century_directory(century_dir)
    empirical_biomass = site['biomass']
    empirical_date = site['date']
    schedule = site['name'] + '.sch'
    schedule_file = os.path.join(input_dir, schedule)
    site_file, weather_file = cent.get_site_weather_files(schedule_file,
                                                          input_dir)
    graz_file = os.path.join(century_dir, "graz.100")
    output = site['name']  # + '_mod-manag'
          
    # check that last block in schedule file includes >= n_months before
    # empirical_date
    cent.check_schedule(schedule_file, n_months, empirical_date)

    # write CENTURY batch file for spin-up simulation
    hist_bat = os.path.join(input_dir, (site['name'] + '_hist.bat'))
    hist_schedule = site['name'] + '_hist.sch'
    hist_output = site['name'] + '_hist'
    cent.write_century_bat(input_dir, hist_bat, hist_schedule, hist_output,
                           fix_file, 'outvars.txt')
    # write CENTURY bat for extend simulation
    extend_bat = os.path.join(input_dir, (site['name'] + '.bat'))
    extend = site['name'] + '_hist'
    output_lis = output + '.lis'
    cent.write_century_bat(input_dir, extend_bat, schedule, output_lis,
                           fix_file, 'outvars.txt', extend)
    # make a copy of the original graz params and schedule file
    shutil.copyfile(graz_file, os.path.join(century_dir, 'graz_orig.100'))
    shutil.copyfile(schedule_file, os.path.join(input_dir, 'sch_orig.sch'))
    
    spin_up_outputs = [site['name'] + '_hist_log.txt',
                       site['name'] + '_hist.lis']
    century_outputs = [output + '_log.txt', output + '.lis', output + '.bin']
    # move CENTURY run files to CENTURY dir
    e_schedule = os.path.join(input_dir, site['name'] + '.sch')
    h_schedule = os.path.join(input_dir, site['name'] + '_hist.sch')
    file_list = [hist_bat, extend_bat, e_schedule, h_schedule, site_file,
                 weather_file]
    if weather_file == 'NA':
        file_list.remove(weather_file)
    for file in file_list:
        shutil.copyfile(file, os.path.join(century_dir,
                                           os.path.basename(file)))
    run_schedule = os.path.join(century_dir, os.path.basename(e_schedule))
    os.remove(hist_bat)
    os.remove(extend_bat)
    # run CENTURY for spin-up                           
    hist_bat_run = os.path.join(century_dir, (site['name'] + '_hist.bat'))
    century_bat_run = os.path.join(century_dir, (site['name'] + '.bat'))
    cent.launch_CENTURY_subprocess(hist_bat_run)

    # save copies of CENTURY outputs, but remove from CENTURY dir
    intermediate_dir = os.path.join(out_dir, 'CENTURY_outputs_spin_up')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    for file in spin_up_outputs:
        shutil.copyfile(os.path.join(century_dir, file),
                        os.path.join(intermediate_dir, file))
        os.remove(os.path.join(century_dir, file))
    # start summary csv
    summary_csv = os.path.join(out_dir, 'modify_management_summary_' +
                               site['name'] + '.csv')
    try:
        with open(summary_csv, 'wb') as summary_file:
            writer = csv.writer(summary_file, delimiter = ',')
            row = ['Iteration', 'Empirical_biomass', 'Simulated_biomass']
            writer.writerow(row)
            for iter in xrange(max_iterations):
                row = [iter]
                row.append(empirical_biomass)
                
                # call CENTURY from the batch file
                cent.launch_CENTURY_subprocess(century_bat_run)

                intermediate_dir = os.path.join(out_dir,
                                     'CENTURY_outputs_iteration%d' % iter)
                if not os.path.exists(intermediate_dir):
                    os.makedirs(intermediate_dir)
                for file in century_outputs:
                    shutil.copyfile(os.path.join(century_dir, file),
                                    os.path.join(intermediate_dir, file))
                    os.remove(os.path.join(century_dir, file))

                # get simulated biomass
                output_file = os.path.join(intermediate_dir, output_lis)
                biomass_df = cent.read_CENTURY_outputs(output_file,
                                                       math.floor(
                                                           empirical_date) - 1,
                                                       math.ceil(
                                                           empirical_date) + 1)
                biomass_df.drop_duplicates(inplace=True)
                if live_or_total == 'live':
                    simulated_biomass = biomass_df.loc[empirical_date, 'aglivc']
                elif live_or_total == 'total':
                    simulated_biomass = biomass_df.loc[empirical_date, 'total']
                else:
                    er = """Biomass output from CENTURY must be specified as
                            'live' or 'total'"""
                    raise Exception(er)
                row.append(simulated_biomass)
                writer.writerow(row)
                
                # check difference between simulated and empirical biomass
                if float(abs(simulated_biomass - empirical_biomass)) <= \
                                                                     threshold:
                    print "Biomass target reached!"
                    break  # success!
                else:
                    # these are the guts of the routine, where either the
                    # grazing parameters file or the schedule file used in the
                    # CENTURY call are changed
                    diff = simulated_biomass - empirical_biomass
                    increase_intensity = 0
                    if diff > 0:  # simulated biomass greater than empirical
                        increase_intensity = 1
                    if vary == 'intensity':
                        # change % biomass removed in grazing parameters file
                        # TODO the grazing level here should reflect what's in
                        # the schedule file
                        success = cent.modify_intensity(increase_intensity,
                                                        graz_file, 'GLP',
                                                        out_dir, site['name']
                                                        + str(iter))
                        if not success:
                            print """intensity cannot be modified, %d
                                     iterations completed""" % (iter - 1)
                            break
                    elif vary == 'schedule':
                        # add or remove scheduled grazing events
                        target_dict = cent.find_target_month(
                                            increase_intensity, run_schedule,
                                            empirical_date, n_months)
                        cent.modify_schedule(run_schedule, increase_intensity,
                                             target_dict, 'GL', out_dir,
                                             site['name'] + str(iter))
                    elif vary == 'both':
                        target_dict = cent.find_target_month(
                                            increase_intensity, run_schedule,
                                            empirical_date, n_months)
                        if target_dict:
                            # there are opportunities to modify the schedule
                            cent.modify_schedule(run_schedule,
                                                 increase_intensity,
                                                 target_dict, 'GL', out_dir,
                                                 site['name'] + str(iter))
                        else:
                            # no opportunities to modify the schedule exist
                            success = cent.modify_intensity(increase_intensity,
                                                            graz_file, 'GL',
                                                            out_dir,
                                                            site['name'] +
                                                            str(iter))
                            if not success:
                                print """intensity cannot be modified, %d
                                         iterations completed""" % (iter - 1)
                                break
                if iter == (max_iterations - 1):
                    print 'maximum iterations performed, target not reached'
    finally:
        # replace modified grazing parameters and schedule files with original
        # files
        os.remove(graz_file)
        shutil.copyfile(os.path.join(century_dir, 'graz_orig.100'), graz_file)
        os.remove(os.path.join(century_dir, 'graz_orig.100'))
        os.remove(schedule_file)
        shutil.copyfile(os.path.join(input_dir, 'sch_orig.sch'),
                        schedule_file)
        files_to_remove = [os.path.join(century_dir, os.path.basename(
                                                       f)) for f in file_list]
        for file in files_to_remove:
            os.remove(file)
        os.remove(os.path.join(input_dir, 'sch_orig.sch'))
        os.remove(os.path.join(century_dir, site['name'] + '_hist.bin'))
