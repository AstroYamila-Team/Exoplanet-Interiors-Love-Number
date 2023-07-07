import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfcinv
from scipy.special import erfc
'''
The cube is a uniform distribution between 0 and 1. These priors all transform that into a relevant prior.
'''


def uniform_prior(x, low_lim, up_lim):
	return (up_lim - low_lim)*x+low_lim
	
	
def GaussianPrior(x,mu,sigma):
	"""Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""

	if (x <= 1.0e-16 or (1.0-x) <= 1.0e-16):
		return -1.0e32
	else:
		return mu+sigma*np.sqrt(2.0)*erfcinv(2.0*(1.0-x)) #I'm sure there is a very mathy reason for why we use this, but I'm just accepting it for now. Let's see if it works first.
		
def GaussianPrior_truncated(x,mu,sigma, low_lim, up_lim):
	"""Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2] with cutoffs at the limits. Prevents extreme values"""
	low_x = 1-0.5*erfc((low_lim-mu)/(sigma*np.sqrt(2.0)))
	up_x = 1-0.5*erfc((up_lim-mu)/(sigma*np.sqrt(2.0)))
	limited_x = (up_x - low_x)*x+low_x
	if (limited_x <= 1.0e-16 or (1.0-limited_x) <= 1.0e-16):
		return -1.0e32
	else:
		return mu+sigma*np.sqrt(2.0)*erfcinv(2.0*(1.0-limited_x))
