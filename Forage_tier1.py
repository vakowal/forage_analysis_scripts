import forage_utils as forage
import time
import os
import errno

############ USER INPUTS ############
workspace_dir = "C:\\Users\\Ginger\\Documents\\Python\\Output"
replicate_runs = 1 
num_years = 1 # total number of years to simulate
cell_size_ha = 5 
cell_number = 1 # how many cells to simulate?
activity_level = 'high'  # acceptable values: 'high', 'moderate', 'low'
time_step = 'month'
steps_per_year = forage.find_steps_per_year(time_step)

initial_conditions = {
    u'standing_veg': 1000.0,
    u'forage_quality': 'low',  # 'grain', 'high', 'moderate', 'low'
    u'herd_size': 16.0, # from Rubanza et al 2005
    u'average_weight_kg': 160.0, # young zebu steer: from Rubanza et al 2005
    u'f_mature_weight_kg': 251.0, # female: from King et al 1984
}
######################################

# Make directory to store summary results and input parameters
current_time = time.strftime("%y%m%d-%H%M")
dir_name = os.path.join(workspace_dir, ('Forage_Model_' + current_time))
try:
    os.makedirs(dir_name)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

# Make parameters array and write input parameter values to output file
parameters = []
parameters.append("Start time: " + current_time)
parameters.append("Directory: " + dir_name)
parameters.append("Replicate runs: " + str(replicate_runs))
parameters.append("Num years: " + str(num_years))
parameters.append("Cell size ha: " + str(cell_size_ha))
parameters.append("Cell number: " + str(cell_number))
parameters.append("Activity level: " + activity_level)
parameters.append("Timestep: " + str(time_step))
parameters.append("Initial standing veg: " + str(initial_conditions['standing_veg']))
parameters.append("Forage quality: " + str(initial_conditions['forage_quality']))
parameters.append("Herd size: " + str(initial_conditions['herd_size']))
parameters.append("Average weight (kg): " + str(initial_conditions['average_weight_kg']))
parameters.append("F mature weight (kg): " + str(initial_conditions['f_mature_weight_kg']))

settings_file = os.path.join(dir_name, 'Settings_tier1.txt')
with open(settings_file, 'w') as out:
    out.write('SETTINGS: FORAGE TIER 1 \n')
    out.write ('-------------------------\n\n')
    for param in parameters:
        out.write(param + '\n')

# Begin summary csv of standing vegetation and herd average weight
summary_csv = open(os.path.join(dir_name, 'summary.csv'), 'w')
summary_csv.write("Year,Step,Standing_veg,Herd_avg_weight,\n0,0,%f,%f\n" %
                 (initial_conditions['standing_veg'],
                 initial_conditions['average_weight_kg']))
######################################

for replicate in range(1, replicate_runs + 1): # for each replicate run
    # initialize replicate placeholders
    veg = forage.VegT1(initial_conditions['standing_veg'], 
                    initial_conditions['forage_quality'])
    herd = forage.HerdT1(
        initial_conditions['herd_size'],
        initial_conditions['average_weight_kg'],
        initial_conditions['f_mature_weight_kg'],
    )
    # get random weather?
    for year in range(1, num_years + 1):
        # get random weather?
        for step in range(1, steps_per_year + 1):
            summary_csv.write("%i, %i" % (year,step))
            # access weather for this time step
            for cell in range(1, cell_number + 1): # for each cell
                ## grow grass ###
                delta_veg = 1960.
                veg.standing_veg += delta_veg # add grass
                veg.available_veg = veg.standing_veg # grass
                    # available for grazing -- here is where we add
                    # constraints to overgrazing
                # calculate energy for maintenance for entire herd
                e_maintenance = herd.e_maintenance()
                # select diet, intake
                diet = veg.diet_selection(herd.average_weight_kg,
                    herd.size)
                # allocate energy from diet among the herd
                herd.e_allocate(diet, e_maintenance,
                                activity_level)
                summary_csv.write(", %f, %f\n" % (veg.standing_veg,
                                  herd.average_weight_kg))
            # end cell loop
        # end time step loop
    # end yearly loop
summary_csv.close()
