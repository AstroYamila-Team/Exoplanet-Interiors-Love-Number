#! /usr/bin/env python3

import os
import time
import _pickle
import numpy as np
from numpy.linalg.linalg import inv
from numpy.lib.twodim_base import diag
from collections import OrderedDict
from importlib import reload

from mpi4py import MPI
import shutil, os

import pyCEPAM_newcode as pyCEPAM
from fits_module_new import fit_function
from astropy.table import Table

# this is for now only the hom_Teq_log_z_ev run
# 2-2-2023: now it also works for dilute_hom_Teq_log_z_ev, dilute_hom_Teq_log_z_y_ev and hom_Teq_log_z_y_ev
# also for with love number


def pymultinest_run(run, radius, sigma_r, j2, sigma_j2, mass, sigma_mass, log_z, log_z_sigma, Teq, Teq_sigma, P, P_sigma, resume=False, lum_up=1.e29, lum_low=1.e24):
    "Function that builds the subdirectories and initilizes the different properties of the planet"
    # setup MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ncpu0 = comm.Get_size()
    if rank==0: print('number of cpus:', ncpu0)
    
    if 'dilute' in run:
        vpar = ['mnoyau', 'yz', 'T0', 'mdilute', 'zdilute']
    else:
        vpar = ['mnoyau', 'yz', 'T0']
    subd = ["{0:02d}".format(n) for n in range(ncpu0)]
    
    if 'k' in run:
        obs = ['req', 'j2']
        values_obs = np.array([radius, j2])
        errors_obs = np.array([sigma_r, sigma_j2])
    else:
        obs = ['req']
        values_obs = np.array([radius])
        errors_obs = np.array([sigma_r])
    
    cepam_config = pyCEPAM.PyCEPAM(
        vpar,'mev',
        cepam_path='/home/s2034174/data1/MRP/clean_prior_planet/dilute/',
        subdirs = (subd),
        mpi=True
    )

    # make the subdirectories for different processes
    if not resume:
        try: 
            cepam_config.remove_tree()
        except:
            pass
        cepam_config.make_tree()
        if rank==0: print('subdirectories are made')

    # go into subdirectory
    if rank < 10:
        os.chdir('0'+str(rank))
    else:
        os.chdir(str(rank))
    
    fit = fit_function(run=run, obs=obs, obj1=cepam_config,\
     values_obs=values_obs, errors_obs=errors_obs, mass=mass, sigma_mass=sigma_mass,\
     log_z=log_z, log_z_sigma=log_z_sigma, Teq=Teq, Teq_sigma=Teq_sigma,\
     P=P, P_sigma=P_sigma,\
     lum_up=lum_up, lum_low=lum_low)
    # print('object is made')
    fit.multinest_fit(resume=resume)

if (__name__ == '__main__'):
    run = 'dilute_Teq_log_z_y_ev'
    run = 'hom_Teq_log_z_k_ev'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ncpu0 = comm.Get_size()

    planets_file = '/home/s2034174/data1/MRP/planets_input.csv'
    planets_info = Table.read(planets_file)
    
    i = 26 # WASP-52_b
    # i = 24 # WASP-39_b
    i = 27 # WASP-6_b
    i = 13 # WASP-103b
    # i = 33 # WASP-77_a_b
    planet_name = planets_info[i]['planet_name']
    planet_path = '/home/s2034174/data1/MRP/'+planet_name+'_testk'
    if rank==0:
        if not os.path.exists(planet_path):
            os.mkdir(planet_path)
    comm.Barrier()
    result_loc = planet_path+'/'+run+'_nolumlim'
    # if rank==0: shutil.copytree('/home/s2034174/data1/MRP/clean_prior_planet/'+run, result_loc)
    comm.Barrier()
    os.chdir(result_loc)
    pymultinest_run(run=run,
    radius=planets_info[i]['radius'], sigma_r=planets_info[i]['sigma_radius'],
    j2=planets_info[i]['J2'], sigma_j2=planets_info[i]['sigma_J2'],
    mass=planets_info[i]['mass'], sigma_mass=planets_info[i]['sigma_mass'], 
    log_z=planets_info[i]['log_z'], log_z_sigma=planets_info[i]['sigma_log_z'],
    Teq=planets_info[i]['Teq'], Teq_sigma=planets_info[i]['sigma_Teq'],
    P=planets_info[i]['P'], P_sigma=planets_info[i]['sigma_P'])
    # lum_up=1.e28, lum_low=1.e27)