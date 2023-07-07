#! /usr/bin/env python3

# import sys
# path_to_files = '/home/s2034174/data1/cepam_runs/modules/'
# sys.path.append(path_to_files)
from emcee_driver import pymultinest_run
from fits_module_new import fit_function
from evolution_runs import lum_run_complete, find_lum_lim_complete

pymultinest_run(run=run,
    radius=planets_info[i]['radius'], sigma_r=planets_info[i]['sigma_radius'],
    j2=planets_info[i]['J2'], sigma_j2=planets_info[i]['sigma_J2'],
    mass=planets_info[i]['mass'], sigma_mass=planets_info[i]['sigma_mass'], 
    log_z=planets_info[i]['log_z'], log_z_sigma=planets_info[i]['sigma_log_z'],
    Teq=planets_info[i]['Teq'], Teq_sigma=planets_info[i]['sigma_Teq'],
    P=planets_info[i]['P'], P_sigma=planets_info[i]['sigma_P'],
    resume=False,
    # lum_up=1.e28, lum_low=1.e27
    )