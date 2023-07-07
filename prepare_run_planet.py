#! /usr/bin/env python3
import shutil, os
from astropy.table import Table
import time
import numpy as np

from evolution_runs import find_lum_lim_complete, read_entropy_luminosity

# change the paths in this file

# specify here which runs you want to run
run = 'dilute_Teq_log_z_y_ev'
run = 'hom_Teq_log_z_ev'
runs = ['hom_Teq_log_z', 'dilute_Teq_log_z']
runs = ['hom_Teq_k_log_z', 'dilute_Teq_k_log_z']


# planets_file = '/home/s2034174/data1/cepam_runs/planets_input.csv'
planets_file = 'planets_love_number.csv'
planets_info = Table.read(planets_file)

# # i = 26 # WASP-52_b
# # # i = 24 # WASP-39_b sanne's values
# # i = 27 # WASP-6_b
# i = 13 # WASP-103b
# # # i = 33 # WASP-77_a_b
# # for i in range(len(planets_info)):
# i = 5 # hat p 13b upper limit
# i = 6 # hat p 13b lower limit
# # i = 4 # wasp-18b
# i = -2 # wasp-39b new metallicity
# # i = -1 # WASP-39b old metallicity
# # i = 0 # hatp13b
# # i = 1 # wasp-4b
# i = 4
# i = -1
# i = 3

jup_rad= 7.1492*10**9
run_planets = [0, 1, 3, 4, 7, 8]
# run_planets = [8]

for i in run_planets:

    planet_name = planets_info[i]['planet_name']
    planet_path = '/home/s2034174/data1/cepam_runs/planets/'+planet_name
    print(planet_name)
    radius = planets_info[i]['radius']
    sigma_r = planets_info[i]['sigma_radius']
    j2 = planets_info[i]['J2']
    sigma_j2 = planets_info[i]['sigma_J2']
    mass = planets_info[i]['mass']
    sigma_mass = planets_info[i]['sigma_mass'] 
    log_z = planets_info[i]['log_z'] 
    log_z_sigma = planets_info[i]['sigma_log_z']
    Teq = planets_info[i]['Teq']
    Teq_sigma = planets_info[i]['sigma_Teq']
    P = planets_info[i]['P'] 
    P_sigma=planets_info[i]['sigma_P']
    age_low = planets_info[i]['age_lowlim']
    age_high = planets_info[i]['age_uplim']

    if not os.path.exists(planet_path):
        shutil.copytree('/home/s2034174/data1/cepam_runs/clean_prior_planet_new/'+run, planet_path)



    lum_dir = 'evolution'

    file = open(planet_path+'/lumrun_planet.py', 'r')
    new_file = ''
    for line in file:
        if line.startswith('lum_run_complete('):
            new_line = "lum_run_complete(subdir=['{}'],".format(lum_dir)+os.linesep
        elif line.startswith('    age_low='):
            new_line = '    age_low={},'.format(age_low)+os.linesep
        elif line.startswith('    age_high='):
            new_line = '    age_high={},'.format(age_high)+os.linesep
        elif line.startswith('    mass='):
            new_line = '    mass={},'.format(mass)+os.linesep
        elif line.startswith('    Teq='):
            new_line = '    Teq={},'.format(Teq)+os.linesep
        else:
            new_line = line
        new_file += new_line
    file.close()

    fout = open(planet_path+'/lumrun_planet.py', 'w')
    fout.write(new_file)
    fout.close()

    if not os.path.exists(planet_path+'/'+lum_dir):
        os.chdir(planet_path)
        os.system('sbatch ./lumrun_planet.slurm')
        os.chdir('/home/s2034174/')
        print(planet_name, ' lumrun submitted')
        time.sleep(3*60)
    try:
        values = find_lum_lim_complete(planet_path+'/'+lum_dir, age_low, age_high, None, full_output=True)
        age, lum, radius_ev = values
        mask = (age>=age_low)&(age<=age_high)
        min_lum = np.min(lum[mask])
        max_lum = np.max(lum[mask])
        radius_up = lum[np.argmin(np.abs(radius_ev - (radius*jup_rad)))]
        print(radius_up)
        print(max_lum)
        print(min_lum)
        if radius_up<max_lum:
            go_on = True
            take_radius_up = False
        else:
            go_on = True
            take_radius_up = False
            radius_up = np.min(lum[radius_ev>((radius+sigma_r)*jup_rad)])
            print(radius_up)
    except FileNotFoundError:
        go_on = False
    # lum, entropy = read_entropy_luminosity(planet_path+'/'+lum_dir)
    # print(entropy)

    ages = [True]
    include_ys = [False, True]
    # ages = [True]

    if go_on:
        if take_radius_up:
            max_lum = radius_up
        for run in runs:
            for age in ages:
                for include_y in include_ys:
                    new_run_name = run
                    if include_y:
                        new_run_name += '_y'
                    if age:
                        if take_radius_up:
                            new_run_name += '_age_lumtest'
                        else:
                            new_run_name += '_age'
                    
                    result_loc = planet_path + '/' + new_run_name
                    if not os.path.exists(result_loc):
                        print(result_loc, 'did not exist yet')
                        shutil.copytree('/home/s2034174/data1/cepam_runs/clean_prior_planet_new/'+run, result_loc)


                    # grid_mass = str(np.round(mass, 1))+'_Mjup'
                    # print(grid_mass)
                    # shutil.copytree('/home/s2034174/data1/cepam_runs/clean_prior_planet_new/'+run, result_loc)
                    # shutil.copytree('/home/s2034174/data1/cepam_runs/grids/homogeneous/'+grid_mass, result_loc)
                    file = open(result_loc+'/run_planet.py', 'r')
                    new_file = ''
                    for line in file:
                        if line.startswith('pymultinest_run'):
                            new_line = "pymultinest_run(run='{}',".format(new_run_name)+os.linesep
                        elif line.startswith('    radius='):
                            new_line = '    radius={}, sigma_r={},'.format(radius, sigma_r)+os.linesep
                        elif line.startswith('    j2='):
                            new_line = '    j2={}, sigma_j2={},'.format(j2, sigma_j2)+os.linesep
                        elif line.startswith('    mass='):
                            new_line = '    mass={}, sigma_mass={},'.format(mass, sigma_mass)+os.linesep
                        elif line.startswith('    log_z='):
                            new_line = '    log_z={}, log_z_sigma={},'.format(log_z, log_z_sigma)+os.linesep
                        elif line.startswith('    Teq='):
                            new_line = '    Teq={}, Teq_sigma={},'.format(Teq, Teq_sigma)+os.linesep
                        elif line.startswith('    P='):
                            new_line = '    P={}, P_sigma={},'.format(P, P_sigma)+os.linesep
                        elif line.startswith('    # lum_up=1.e28, lum_low=1.e27'):
                            if age:
                                new_value = max_lum
                                low_lim = 1.e24
                            else:
                                new_value = 1.e29
                                low_lim = 1.e24
                            new_line = '    lum_up={}, lum_low={}'.format(new_value, low_lim)+os.linesep
                        else:
                            new_line = line
                        new_file += new_line
                    file.close()

                    # # # later add condition for luminosity limits

                    fout = open(result_loc+'/run_planet.py', 'w')
                    fout.write(new_file)
                    fout.close()

        os.chdir(result_loc)
        os.system('sbatch ./run_planet.slurm')
        os.chdir('/home/s2034174/')
        print(planet_name, 'submitted')
