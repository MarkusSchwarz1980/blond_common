# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:01:25 2019

@author: schwarz
"""

#import sys
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append('./../../../')
#import os
#print(os.listdir('./../../../'))
#from blond_common.fitting.distributions import Gaussian
#from blond_common.fitting.distribution import gaussianFit, gaussianFit2

#from .fitting.distributions import Gaussian
from blond_common.fitting.distributions import Gaussian
#import blond_common.devtools.BLond_Rc

time_array = np.arange(0, 25e-9, 0.1e-9)

amplitude = 1.
position = 13e-9
length = 2e-9  # RMS
FWHM = 2*np.sqrt(np.log(4))*length
initial_params_gauss = [amplitude, position, length]
gaussian_dist = Gaussian(*initial_params_gauss)
gaussian_profile = gaussian_dist.profile(time_array)
sigma_gauss = gaussian_dist.RMS

print(gaussian_dist.profile(position+FWHM/2))

bunch_position_gauss, bunch_length_gauss, fitparams_gauss = gaussianFit(
    time_array, gaussian_profile)

print(fitparams_gauss)

fitparams_gauss = gaussianFit2(time_array, gaussian_profile)

print(fitparams_gauss)
gaussian_dist_fit = Gaussian(*fitparams_gauss)
print(gaussian_dist_fit.profile(position+FWHM/2))


