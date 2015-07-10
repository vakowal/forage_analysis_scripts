## guess management by matching biomass in CENTURY
import forage_century_link_utils as cent
import math
import csv
import os
from subprocess import Popen
import shutil

n_years = 2 # how many years to potentially manipulate?
century_dir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Century46_PC_Jan-2014'
graz_file = os.path.join(century_dir, "graz.100")
out_dir = "C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\CENTURY4.6\Kenya\modify_management"  # where to put results of this routine
vary = 'both' # 'schedule', 'intensity', 'both'
live_or_total = 'total'  # 'live' (live biomass) or 'total' (live + standing dead biomass)
threshold = 15.0  # must match biomass within this many g per sq m
max_iterations = 20  # number of times to try
fix_file = 'drytrpfi.100'
outvars = 'outvars.txt'

GT_site = {'name': 'GT', 'biomass': 452.1, 'date': 2012.5}
M05_site = {'name': 'M05', 'biomass': 103.1, 'date': 2013.5}
MO_site = {'name': 'MO', 'biomass': 165.4, 'date': 2012.5}
N4_site = {'name': 'N4', 'biomass': 464.5, 'date': 2014.}
W3_site = {'name': 'W3', 'biomass': 148.8, 'date': 2014.}
W06_site = {'name': 'W06', 'biomass': 315.0, 'date': 2013.5}

site_list = [GT_site, M05_site, MO_site, N4_site, W3_site, W06_site]

for site in site_list:
    empirical_biomass = site['biomass']
    empirical_date = site['date']
    schedule = site['name'] + '.sch'
    century_bat = site['name'] + '.bat'
    output = site['name'] + 'mod-manag.lis'
    schedule_file = os.path.join(century_dir, schedule)
    output_file = os.path.join(century_dir, output)
          
    # check that last block in schedule file includes >= n_years before empirical_date
    cent.check_schedule(schedule_file, n_years, empirical_date)

    # write batch file to call CENTURY
    cent.write_century_bat(century_dir, century_bat, schedule, output, fix_file, outvars)

    # make a copy of the original graz params and schedule file
    shutil.copyfile(graz_file, os.path.join(century_dir, 'graz_orig.100'))
    shutil.copyfile(schedule_file, os.path.join(century_dir, 'sch_orig.100'))

    # start summary csv
    summary_csv = os.path.join(out_dir, 'modify_management_summary' + site['name'] + '.csv')
    try:
        with open(summary_csv, 'wb') as summary_file:
            writer = csv.writer(summary_file, delimiter = ',')
            row = ['Iteration', 'Empirical_biomass', 'Simulated_biomass']
            writer.writerow(row)
            for iter in xrange(max_iterations):
                row = [iter]
                row.append(empirical_biomass)
                
                # call CENTURY from the batch file
                p = Popen(["cmd.exe", "/c " + century_bat], cwd = century_dir)
                p.wait()
                
                # check that CENTURY completed successfully
                if output[-4:] == '.lis':
                    output_log = os.path.join(century_dir, output[:-4] + '_log.txt')
                else:
                    output_log = os.path.join(century_dir, output + '_log.txt')
                cent.check_CENTURY_log(output_log)

                # get simulated biomass
                biomass_df = cent.read_CENTURY_outputs(output_file, math.floor(empirical_date) - 1, math.ceil(empirical_date) + 1)

                if live_or_total == 'live':
                    simulated_biomass = biomass_df.loc[empirical_date, 'aglivc']
                elif live_or_total == 'total':
                    simulated_biomass = biomass_df.loc[empirical_date, 'total']
                else:
                    er = "Biomass output from CENTURY must be specified as 'live' or 'total'"
                    raise Exception(er)
                row.append(simulated_biomass)
                writer.writerow(row)
                
                # check difference between simulated and empirical biomass
                if float(abs(simulated_biomass - empirical_biomass)) <= threshold:
                    print "Biomass target reached!"
                    break  # success!
                else:
                    # these are the guts of the routine, where either the grazing
                    # parameters file or the schedule file used in the CENTURY call
                    # are changed
                    diff = simulated_biomass - empirical_biomass
                    increase_intensity = 0
                    if diff > 0:  # simulated biomass greater than empirical
                        increase_intensity = 1
                    if vary == 'intensity':
                        # change % biomass removed in grazing parameters file
                        # TODO the grazing level here should reflect what's in the
                        # schedule file
                        success = cent.modify_intensity(increase_intensity, graz_file,
                                        'GH', out_dir, site['name'] + str(iter))
                        if not success:
                            print 'intensity cannot be modified, %d iterations completed' % (iter - 1)
                            break
                    elif vary == 'schedule':
                        # add or remove scheduled grazing events
                        target_dict = cent.find_target_month(increase_intensity,
                            schedule_file, empirical_date, n_years)
                        cent.modify_schedule(schedule_file, increase_intensity,
                            target_dict, 'GL', out_dir, site['name'] + str(iter))
                    elif vary == 'both':
                        target_dict = cent.find_target_month(increase_intensity,
                            schedule_file, empirical_date, n_years)
                        if target_dict:
                            # there are opportunities to modify the schedule
                            cent.modify_schedule(schedule_file, increase_intensity,
                                    target_dict, 'GL', out_dir, site['name'] + str(iter))
                        else:
                            # no opportunities to modify the schedule exist
                            success = cent.modify_intensity(increase_intensity, graz_file,
                            'GL', out_dir, site['name'] + str(iter))
                            if not success:
                                print 'intensity cannot be modified, %d iterations completed' % (iter - 1)
                                break
                if iter == (max_iterations - 1):
                    print 'maximum iterations performed, target not reached'
    finally:
        # replace modified grazing parameters and schedule files with original files
        os.remove(graz_file)
        shutil.copyfile(os.path.join(century_dir, 'graz_orig.100'), graz_file)
        os.remove(os.path.join(century_dir, 'graz_orig.100'))
        os.remove(schedule_file)
        shutil.copyfile(os.path.join(century_dir, 'sch_orig.100'), schedule_file)
        os.remove(os.path.join(century_dir, 'sch_orig.100'))
        
