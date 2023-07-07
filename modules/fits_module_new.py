import numpy as np
import matplotlib.pyplot as plt
import pymultinest
import os, threading, subprocess
if not os.path.exists("chains"): os.mkdir("chains") # This is important.
# This makes a directory called "chains". This is where the walker information
# is being stored by Multinest. All Pymultinest does is then read them in.
import time
from priors_mn import *
print(os.getcwd())
import scipy.special as special

from astropy.io import fits
import astropy.constants as const
import astropy.units as u
import emcee
import scipy.optimize as opt
from astropy.table import Table
import corner
from mpi4py import MPI

print('Version 21-04-2023 15:07')

class fit_function():
    def __init__(self, run, obs, obj1, values_obs, errors_obs, mass,\
     sigma_mass, log_z, log_z_sigma,  Teq, Teq_sigma, P, P_sigma, lum_up=1.e29, lum_low=1.e24):
        self.jup_mass=1.89813*10**30
        self.jup_rad= 7.1492*10**9
        self.jup_lum = 3.35e24 # (8.67e-10 * const.L_sun).to(u.erg/u.s).value
        # print(jup_lum)#.value()
        self.y = values_obs #np.array([radius*jup_rad])
        self.y[0] = self.y[0] * self.jup_rad
        self.yerr = errors_obs #np.array([sigma_r*jup_rad])
        self.yerr[0] = self.yerr[0] * self.jup_rad
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.obj1 = obj1
        self.o = obs
        self.run = run
        if 'y' in self.run:
            y_mu = 0.277
            sigma_y = 0.01
        else:
            y_mu = 0.0
            sigma_y = 0.0
        if log_z is None:
            log_z=0
            log_z_sigma=0
        if (log_z==0 and log_z_sigma==0):
            self.gaussian_z = False
        else:
            self.gaussian_z = True
        
        # print(self.obj1)
        self.low_lims = [5.e28/self.jup_mass, 0.0, 0.0, -12., lum_low/self.jup_lum, 0.0, 0.0]
        self.up_lims = [20.e30/self.jup_mass, 1.0, 1.0, -0.1, lum_up/self.jup_lum, 6000.0, 3.e6]
        self.mus = [mass, 0.0, y_mu, log_z, 0.0, Teq, P]
        self.sigmas = [sigma_mass, 0.0, sigma_y, log_z_sigma, 0.0, Teq_sigma, P_sigma]
        self.parameters = ['Mass', 'mcore', 'y', 'z', 'lum', 'Teq', 'P']
        # if 'k' in self.run:
        #     self.low_lims.extend([0.0])
        #     self.up_lims.extend([3.e6])
        #     self.mus.extend([P])
        #     self.sigmas.extend([P_sigma])
        if 'dilute' in self.run:
            self.low_lims.extend([0.0, 0.0])
            self.up_lims.extend([1.0, 1.0])
            self.mus.extend([0.0, 0.0])
            self.sigmas.extend([0.0, 0.0])
            self.parameters.extend(['mdilute', 'zdilute'])

        self.ndim=len(self.mus)
        # for homogeneous: [mass, mcore, y, log_z, lum_int, Teq, P]
        # for dilute: [mass, mcore, y, log_z, lum_int, Teq, P, mdilute, zdilute]

    
    def function_hom(self, mass, mcore, y, z, lum, Teq, P):
        p = [mcore, y, z, 0.0, 1000.]
        o = self.o
        cepam_obj = self.obj1 
        cepam_obj.mass_value = mass*self.jup_mass
        cepam_obj.Teq_value = Teq
        cepam_obj.lum_value = lum*self.jup_lum
        cepam_obj.P_value = P
        try:
            omod, ierr = cepam_obj.model_mult_MPI(p, o)
            dict_result = cepam_obj.get_obs(o)
            obs_values = np.array([value for value in dict_result.values()])
        except:
            obs_values = np.zeros(len(o))
        print(obs_values)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        ncpu0 = comm.Get_size()
        rank_str="{0:02d}".format(rank)

        output = np.concatenate((obs_values, np.array([mass, mcore, y, z, lum, Teq, P])))
        str_output = np.concatenate((o, np.array(['mass', 'mcore', 'y', 'z', 'lum_int', 'Teq', 'P'])))
        try:
            for i, name in enumerate(str_output):
                previous = np.load('./'+name+'.npy')
                new = np.append(previous, output[i])
                np.save('./'+name+'.npy', new)
        except:
            for i, name in enumerate(str_output):
                new = np.array(output[i])
                np.save('./'+name+'.npy', new)
        
        cepam_obj.clean()
        return obs_values

    def function_dilute(self, mass, mcore, y, z, lum, Teq, P, mdilute, zdilute):
        p = [mcore, y, z, 0.0, 1000., mdilute, zdilute, 0.0]
        o = self.o
        cepam_obj = self.obj1 
        cepam_obj.mass_value = mass*self.jup_mass
        cepam_obj.Teq_value = Teq
        cepam_obj.lum_value = lum*self.jup_lum
        cepam_obj.P_value = P
        try:
            omod, ierr = cepam_obj.model_mult_MPI(p, o)
            dict_result = cepam_obj.get_obs(o)
            obs_values = np.array([value for value in dict_result.values()])
        except:
            obs_values = np.zeros(len(o))
        print(obs_values)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        ncpu0 = comm.Get_size()
        rank_str="{0:02d}".format(rank)

        output = np.concatenate((obs_values, np.array([mass, mcore, y, z, lum, Teq, P, mdilute, zdilute])))
        str_output = np.concatenate((o, np.array(['mass', 'mcore', 'y', 'z', 'lum_int', 'Teq', 'P', 'mdilute', 'zdilute'])))
        try:
            for i, name in enumerate(str_output):
                previous = np.load('./'+name+'.npy')
                new = np.append(previous, output[i])
                np.save('./'+name+'.npy', new)
        except:
            for i, name in enumerate(str_output):
                new = np.array(output[i])
                np.save('./'+name+'.npy', new)
        
        cepam_obj.clean()
        return obs_values

    def model(self, params): 
        theta=[params[i] for i in range(self.ndim)]
        if 'hom' in self.run:
            return self.function_hom(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])
        if 'dilute' in self.run:
            return self.function_dilute(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8])
        else:
            print('That run is not implemented')
    
    def newprior(self, cube, ndim, n_params):
        low_lims=self.low_lims
        up_lims=self.up_lims
        mus=self.mus
        sigmas=self.sigmas
        # [mass, mcore, y, z, lum_int, Teq, P, mdilute, zdilute]
        if 'hom' in self.run:
            if 'y' in self.run:
                uniform_sel = [1, 4]
            else:
                uniform_sel = [1, 2, 4]
        if 'dilute' in self.run:
            if 'y' in self.run:
                uniform_sel = [1, 4, 7, 8]
            else:
                uniform_sel = [1, 2, 4, 7, 8]
        if not self.gaussian_z:
            uniform_sel.append(3)
            low_lims[3] = 0.0
            up_lims[3] = 1.0
        for i in range(ndim):
            if i in uniform_sel: 
                cube[i] = uniform_prior(cube[i], low_lims[i], up_lims[i])
            else:
                cube[i]= GaussianPrior_truncated(cube[i],mus[i],sigmas[i], low_lims[i], up_lims[i])
                #cube[i]= GaussianPrior(cube[i],mus[i],sigmas[i])
        if self.gaussian_z:
            cube[3]=10**cube[3]
        
    def myloglike(self, cube, ndim, n_params): # removed self.x
        '''The most important function. What is your likelihood function?
        This is simple chi2 gaussian errors likelihood here. It assumes
        the measurements are independent and uncertainties are Gaussian.'''
        if (cube[2]+cube[3])>0.75:
            return -np.inf
        # if cube[8]<cube[3]: # <- zdilute < y ????? (constraint on z_dilute?)
        #     return -np.inf
        loglike = -np.sum(((self.y-self.model(cube))/self.yerr)**2 + np.log(2*np.pi*np.power(self.yerr,2)))
        return loglike/2.

    def multinest_fit(self, resume=False, verbose=False):
        #print('Doing a multinest fit')
        ndim = self.ndim
        n_params = ndim 
        print(os.getcwd())
        pymultinest.run(self.myloglike, self.newprior, n_params, 
            #resume = False, verbose = True, sampling_efficiency = 0.8, n_live_points=50)
            resume = resume, verbose = True, sampling_efficiency = 0.3, n_live_points=400)
            #resume = False, verbose = True, sampling_efficiency = 0.8, n_live_points=400)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank==0:
            a = pymultinest.Analyzer(n_params = n_params)
            s = a.get_stats()
            values = s['marginals']
            self.evidence = s['global evidence']
            self.ev_sigma = s['global evidence error']

            print("-" * 30, 'ANALYSIS', "-" * 30)
            print( "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] ))
            parameters=[]
            print( 'Parameter values with 1 sigma uncertainty:')
            for i in range(self.ndim):
                print( str(i+1)+':',values[i]['median'],'pm',values[i]['sigma'])
                parameters =np.append(parameters, str(i+1))
            self.fit_values = [values[i]['median'] for i in range(ndim)]
            print(self.fit_values)
            self.fit_sigmas = [values[i]['sigma'] for i in range(ndim)]
            print('creating marginal plot ...')
            data = a.get_data()[:,2:]
            weights = a.get_data()[:,0]

            #mask = weights.cumsum() > 1e-5
            mask = weights > 1e-4
            print(self.parameters)
            corner.corner(data[mask,:], weights=weights[mask], 
                labels=self.parameters, show_titles=True)
            plt.savefig('corner.pdf')
            plt.savefig('corner.png')
            plt.close()
