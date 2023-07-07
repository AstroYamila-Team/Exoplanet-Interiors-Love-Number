import pyCEPAM_newcode as pycepam
# import pymultinest
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def replace_agemax(value):
    file = open('cepam.don', 'r')
    new_file = ''
    for line in file:
        if line.startswith(' AGEMAX ='):
            new_line = 'AGEMAX =  {},'.format(value)+os.linesep
        else:
            new_line = line
        new_file += new_line
    file.close()
    
    fout = open('cepam.don', 'w')
    fout.write(new_file)
    fout.close()
    return

def lum_run(subdir, age_low, mass, Teq, mcore, y, z, max_rows, mode='maximum', full_output=False):
    if mode=='minimum2':
        mode='minimum'
    if mode=='minimum':
        bound = 'lum_min'
        minimum = True
    if mode=='maximum':
        bound = 'lum_max'
        minimum = False
    if mode=='normal':
        minimum = False
    print('minimum=', minimum)
    # add subdir copy tree enz.
    if max_rows != None:
        if max_rows>0:
            max_rows = int(max_rows)
        else:
            max_rows=None
        print('max_rows:', max_rows)
    else:
        max_rows = None
        print('max_rows:', max_rows)
    jup_mass=1.89813*10**30
    jup_rad= 7.1492*10**9

    p = [mcore, y, z, 0.0]
    vpar = ['mnoyau', 'yz']
    cepam_obj = pycepam.PyCEPAM(vpar=vpar, pfx='mev', cepam_path='/home/s2034174/data1/MRP/clean_prior_planet/dilute/', subdirs=subdir)
    
    try:
        cepam_obj.remove_tree()
    except:
        pass
    cepam_obj.make_tree()
    if subdir != None:
        os.chdir(subdir[0])
    cepam_obj.mass_value = mass * jup_mass
    cepam_obj.Teq_value = Teq
    values, ierr = cepam_obj.evolution_model(p, age_low, max_rows, minimum, full_output)
    # if subdir != None:
    #     os.chdir('..')
    if mode=='normal':
        return
    elif full_output:
        return values[0], values[1]
    else:
        print(values)
        return values[bound]

def find_lum_lim(path, max_rows, age_lim, mode='maximum', full_output=False):
    if max_rows != None:
        if max_rows>0:
            print(max_rows)
            max_rows = int(max_rows)
        else:
            max_rows=None
            print(max_rows)
    else:
        max_rows = None
    output = np.loadtxt(path+'/mev_ev.csv', delimiter=',', skiprows=1, max_rows=None)
    age = output[:, 0]
    lum = output[:, 2]
    radius = output[:, 1]
    mask=(age>=age_lim)
    if mode=='minimum' or mode=='minimum2':
        mask = (age<=age_lim)
        try: 
            lum_min = np.min(lum[mask])
        except:
            lum_min = np.min(lum)
            print('lum does not correspond to age')
        return 
    elif full_output:
        return age, lum, radius
    else:
        try:
            lum_max = np.max(lum[mask])
        except:
            lum_max = np.min(lum) # why min?
            print('lum does not correspond to age')
        return lum_max

def find_lumrun_input(path_result):
    a = pymultinest.Analyzer(n_params=6,\
    outputfiles_basename=path_result+'/00/chains/1-')
    s = a.get_stats()
    values = s['marginals']
    mcore_new = values[1]['median']
    y_new = values[2]['median']
    z_new = values[3]['median']
    print('mcore, y, z found:', mcore_new, ',', y_new, ',', z_new)
    return mcore_new, y_new, z_new

def lum_run_complete_dilute(subdir, age_low, age_high, mass, Teq, mcore, mdilute, zdilute, y, z, max_rows, radius_plot=False):
    # make sure max_rows has value or None
    if max_rows != None:
        if max_rows>0:
            max_rows = int(max_rows)
        else:
            max_rows=None
        print('max_rows:', max_rows)
    else:
        max_rows = None
        print('max_rows:', max_rows)
    
    jup_mass=1.89813*10**30
    jup_rad= 7.1492*10**9
    
    if age_high > 10**4:
        replace_agemax(value='1.d5')
    
    p = [mcore, y, z, 0.0, 1000., mdilute, zdilute, 0.0]
    vpar = ['mnoyau', 'yz', 'T0', 'mdilute', 'zdilute']
    cepam_obj = pycepam.PyCEPAM(vpar=vpar, pfx='mev', cepam_path='/home/s2034174/data1/MRP/clean_prior_planet/dilute/', subdirs=subdir)
    
    try:
        cepam_obj.remove_tree()
    except:
        pass
    cepam_obj.make_tree()
    if subdir != None:
        os.chdir(subdir[0])
    cepam_obj.mass_value = mass * jup_mass
    cepam_obj.Teq_value = Teq
    values, ierr = cepam_obj.evolution_model(p, age_low, max_rows, minimum=False, full_output=True)
    age, lum = values

    fig = plt.figure(figsize=[7, 4])
    plt.xlabel('age (Myr)')
    plt.ylabel(r'$L_{int}$ (erg $s^{-1}$)')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(age, lum, label=f'mcore={mcore}, y={y}, z={z}')
    plt.legend()
    plt.axvline(age_low, ymin=0, ymax=1, linestyle='dashed', color='black')
    plt.axvline(age_high, ymin=0, ymax=1, linestyle='dashed', color='black')
    plt.savefig('evolution.png')

    mask=(age>=age_low)&(age<=age_high)
    max_lum = np.max(lum[mask])
    min_lum = np.min(lum[mask])

    return values, min_lum, max_lum

def read_entropy_luminosity(subdir):
    data = np.loadtxt(subdir+'/mev_ev.csv', delimiter=',', skiprows=1)

    luminosity = data[:, 2]
    entropy = data[:, 7]

    k_b = 1.380649e-16 # erg/K
    m_H = 1.6737236e-24
    m_He = 6.646442e-24
    ratio_H = 110/(110+9)
    ratio_He = 1. - ratio_H

    entropy_kb_e = entropy/k_b * (ratio_H * m_H + ratio_He * m_He)
    return luminosity, entropy_kb_e


def lum_run_complete(subdir, age_low, age_high, mass, Teq, mcore, y, z, max_rows, radius_plot=False):
    # make sure max_rows has value or None
    if max_rows != None:
        if max_rows>0:
            max_rows = int(max_rows)
        else:
            max_rows=None
        print('max_rows:', max_rows)
    else:
        max_rows = None
        print('max_rows:', max_rows)
    
    jup_mass=1.89813*10**30
    jup_rad= 7.1492*10**9
    
    if age_high > 10**4:
        replace_agemax(value='1.d5')
    
    p = [mcore, y, z, 0.0]
    vpar = ['mnoyau', 'yz']
    cepam_obj = pycepam.PyCEPAM(vpar=vpar, pfx='mev', cepam_path='/home/s2034174/data1/MRP/clean_prior_planet/dilute/', subdirs=subdir)
    
    try:
        cepam_obj.remove_tree()
    except:
        pass
    cepam_obj.make_tree()
    if subdir != None:
        os.chdir(subdir[0])
    cepam_obj.mass_value = mass * jup_mass
    cepam_obj.Teq_value = Teq
    values, ierr = cepam_obj.evolution_model(p, age_low, max_rows, minimum=False, full_output=True)
    print(values, ierr)
    age, lum = values

    output = np.loadtxt('mev_ev.csv', delimiter=',', skiprows=1, max_rows=max_rows)
    radius = output[:, 1]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=[7, 4])
    ax[1].set_xlabel('age (Myr)')
    ax[0].set_ylabel(r'$L_{int}$ (erg $s^{-1}$)')
    ax[1].set_ylabel(r'Radius ($R_J)$')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].yaxis.set_major_formatter(ScalarFormatter())

    ax[0].plot(age, lum)
    ax[1].plot(age, radius/jup_rad)#, label=f'mcore={mcore}, y={y}, z={z}')
    
    ax[1].axvline(age_low, ymin=0, ymax=1, linestyle='dashed', color='black', label='stellar age limits')
    ax[0].axvline(age_high, ymin=0, ymax=1, linestyle='dashed', color='black')
    ax[0].axvline(age_low, ymin=0, ymax=1, linestyle='dashed', color='black')
    ax[1].axvline(age_high, ymin=0, ymax=1, linestyle='dashed', color='black')
    # ax[1].axhline(1.14, linestyle='dotted', color='black', label='observed radius')
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.tight_layout()
    plt.savefig('evolution.png', bbox_inches='tight')

    if radius_plot:
        output = np.loadtxt('mev_ev.csv', delimiter=',', skiprows=1, max_rows=max_rows)
        radius = output[:, 1]
        fig = plt.figure(figsize=[7, 4])
        plt.xlabel(r'Radius ($R_J$)')
        plt.ylabel(r'$L_{int}$ (erg $s^{-1}$)')
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(radius/jup_rad, lum, label=f'mcore={mcore}, y={y}, z={z}')
        plt.legend()
        plt.savefig('evolution_radius.png')

    mask=(age>=age_low)&(age<=age_high)
    max_lum = np.max(lum[mask])
    min_lum = np.min(lum[mask])

    if radius_plot:
        values = age, lum, radius

    return values, min_lum, max_lum

def find_lum_lim_complete(path, age_low, age_high, max_rows, full_output=False):
    if max_rows != None:
        if max_rows>0:
            print(max_rows)
            max_rows = int(max_rows)
        else:
            max_rows=None
            print(max_rows)
    else:
        max_rows = None
    output = np.loadtxt(path+'/mev_ev.csv', delimiter=',', skiprows=1, max_rows=None)
    age = output[:, 0]
    lum = output[:, 2]
    radius = output[:, 1]
    mask=(age>=age_low)&(age<=age_high)
    if full_output:
        return age, lum, radius
    else:
        return np.min(lum[mask]), np.max(lum[mask])

def find_age(path, luminosity):
    output = np.loadtxt(path+'/mev_ev.csv', delimiter=',', skiprows=1, max_rows=None)
    age = output[:, 0]
    lum = output[:, 2]
    difference = np.abs(lum - luminosity)
    # print(difference)
    minimum = np.min(difference)
    index = np.argwhere(difference == minimum)
    return age[index]