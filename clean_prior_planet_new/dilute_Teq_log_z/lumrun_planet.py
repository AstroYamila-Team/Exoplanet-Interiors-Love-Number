#! /usr/bin/env python3

from emcee_driver import pymultinest_run
from fits_module_new import fit_function
from evolution_runs import lum_run_complete_dilute, find_lum_lim_complete

lum_run_complete_dilute(subdir=['lum_run'],
    age_low=planets_info[i]['age_lowlim'],
    age_high=planets_info[i]['age_uplim'],
    mass=planets_info[i]['mass'],
    Teq=planets_info[i]['Teq'],
    mcore=0.05,
    mdilute=0.27,
    zdilute=0.2,
    y=0.3,
    z=0.5,
    max_rows=None)