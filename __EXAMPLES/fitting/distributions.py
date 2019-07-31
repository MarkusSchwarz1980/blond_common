#!/usr/bin/env python
# coding: utf-8

# # Using the fitting module to fit distributions (e.g. bunch line density)

# In[1]:


# Adding folder on TOP of blond_common to PYTHONPATH
import sys
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
sys.path.append('./../../../')


# ## 0. Defining a distribution to fit

# The line densities are defined as follow:
# 
# Gaussian:
# $\lambda = \lambda_0 e^{-\frac{\left(t-t_0\right)^2}{2\sigma^2}}$
# 
# Binomial:
# $\lambda = \lambda_0 \left[1-4\left(\frac{t-t_0}{\tau_l}\right)^2\right]^\mu$
# with $\sigma=\frac{\tau_l}{2\sqrt{3+2\mu}}$ ("Parabolic amplitude" is $\mu=3/2$)

# In[2]:


# Here we generate some distributions to test the fitting routines
# NB: the parameters for all functions are organized as [amplitude, position, length, exponent]

from blond_common.fitting.distributions import Gaussian, BinomialAmplitudeN
#from blond_common.fitting.distribution_functions import _binomialRMS

time_array = np.arange(0, 25e-9, 0.1e-9)

amplitude = 1.
position = 13e-9
length = 2e-9
initial_params_gauss = [amplitude, position, length]
gaussian_dist = Gaussian(*initial_params_gauss)
gaussian_profile = gaussian_dist.profile(time_array)
sigma_gauss = gaussian_dist.RMS

amplitude = 0.77
position = 18.3e-9
length = 3.45e-9
exponent = 3.4
initial_params_binom = [amplitude, position, length, exponent]
binom_dist = BinomialAmplitudeN(*initial_params_binom)
binom_profile = binom_dist.profile(time_array)
sigma_binom = binom_dist.RMS


# In[3]:


plt.figure('Distributions')
plt.clf()
plt.plot(time_array*1e9, gaussian_dist.profile(time_array), label='Gaussian')
plt.plot(time_array*1e9, binom_dist.profile(time_array), label='Binomial with $\\mu=3.4$')
plt.xlabel('Time [ns]')
plt.legend(loc='best')


# ## 1. RMS related functions

# In[4]:


# The RMS parameters can be obtained directly from the line density using the RMS function

from blond_common.fitting.distribution import RMS

rms_gauss = RMS(time_array, gaussian_profile)
rms_binom = RMS(time_array, binom_profile)

print('Gauss: Input ->',[initial_params_gauss[1], sigma_gauss], '/ Output ->', rms_gauss[0:2])
print('Binomial: Input ->',[initial_params_binom[1], sigma_binom], '/ Output ->', rms_binom[0:2])


# ## 2. Full-Width Half Maximum related functions

# In[5]:


# The FWHM can be obtained, note that the level can be manually set

from blond_common.fitting.distribution import FWHM, PlotOptions

plotOpt=PlotOptions(figname='FWHM', clf=False)

fwhm_gauss = FWHM(time_array, gaussian_profile, plotOpt=plotOpt)
fwXm_binom = FWHM(time_array, binom_profile, level=0.2, plotOpt=plotOpt)


# In[6]:


# The FWHM can be obtained, note that the level can be manually set
# The bunchLengthFactor option can be used to rescale the FWHM to another value
# e.g. : to 4sigma assuming Gaussian, or parabolic_line, or parabolic_amplitude

from blond_common.fitting.distribution import FWHM, FitOptions

fitOpt = FitOptions(bunchLengthFactor='gaussian')
fwhm_gauss = FWHM(time_array, gaussian_profile, fitOpt=fitOpt)

print('Gauss: Input ->',[initial_params_gauss[1], gaussian_dist.fourSigma_FWHM], '/ Output ->', fwhm_gauss[0:2])



# In[7]:


# The width at 2 different levels can give all needed information on the binomial parameters
# The binomialParametersFromRatio function can be used directly

from blond_common.fitting.distribution import binomialParametersFromRatio, PlotOptions

plotOpt=PlotOptions(figname='BinomRatio', clf=False)

binom_params_binom = binomialParametersFromRatio(time_array, binom_profile, plotOpt=plotOpt)

print('Binomial: Initial ->',initial_params_binom, '/ Final ->', binom_params_binom[-1])


# In[8]:


# The rms bunch lengths as input and using binomialParametersFromRatio can be compared

from blond_common.fitting.distribution import binomialParametersFromRatio
from blond_common.fitting.distribution_functions import _binomialRMS

binom_params_binom = binomialParametersFromRatio(time_array, binom_profile)

print('Binomial: Initial ->',sigma_binom, '/ Final ->', _binomialRMS(*binom_params_binom[-1]))


# In[9]:


# For the Gaussian case, the exponent goes to infinity so a fair
# approximation consists of extending the look-up table for the binomial parameters

from blond_common.fitting.distribution import PlotOptions
from blond_common.fitting.distribution import binomialParametersFromRatio

plotOpt=PlotOptions(figname='BinomRatio-2', clf=False)

binom_params_gauss = binomialParametersFromRatio(time_array, gaussian_profile, plotOpt=plotOpt)

print('Gauss: Initial ->',initial_params_gauss, '/ Final ->', binom_params_gauss[-1])


# In[10]:


# For the Gaussian case, the exponent goes to infinity so a fair
# approximation consists of extending the look-up table for the binomial parameters

from blond_common.fitting.distribution import PlotOptions
from blond_common.fitting.distribution import binomialParametersFromRatio, _binomialParametersFromRatioLookupTable

plotOpt=PlotOptions(figname='BinomRatio-3', clf=False)

newLookupTable = _binomialParametersFromRatioLookupTable(
    exponentMin=100, exponentMax=10000)

binom_params_gauss = binomialParametersFromRatio(
    time_array, gaussian_profile, plotOpt=plotOpt,
    ratioLookUpTable=newLookupTable)

print('Gauss: Initial ->',initial_params_gauss, '/ Final ->', binom_params_gauss[-1])


# ## 3. Distribution fitting routines

# In[11]:


from blond_common.fitting.distribution import gaussianFit, binomialAmplitudeNFit, PlotOptions

bunch_position_gauss, bunch_length_gauss, fitparams_gauss = gaussianFit(
    time_array, gaussian_profile)

bunch_position_binom, bunch_length_binom, fitparams_binom = binomialAmplitudeNFit(
    time_array, binom_profile)

#TODO fix different convention of exponent mu!
print('Gauss: Initial ->',initial_params_gauss, '/ Final ->', fitparams_gauss)
print('Binomial: Initial ->',initial_params_binom, '/ Final ->', fitparams_binom)


# In[12]:


plotOpt=PlotOptions(figname='Fit-1', clf=False)

bunch_position_binom, bunch_length_binom, fitparams_binom = binomialAmplitudeNFit(
    time_array, binom_profile, plotOpt=plotOpt)


# In[13]:


# Using a fit function that does not necessarily correspond to the input
# (e.g. parabolicAmplitudeFit on a binomial distribution with more tails)

from blond_common.fitting.distribution import gaussianFit, parabolicAmplitudeFit, binomialAmplitudeNFit, PlotOptions

plotOpt=PlotOptions(figname='Fit-2', clf=False)

parabolicAmplitudeFit(time_array, binom_profile, plotOpt=plotOpt)


# In[14]:


# Using custom FitOptions
# NB: a new FitOptions should be created to reset initial conditions,
# but the same FitOptions can be used to share the same initial conditions

from blond_common.fitting.distribution import gaussianFit, parabolicAmplitudeFit, binomialAmplitudeNFit, FitOptions

fitOpt = FitOptions(fittingRoutine='minimize')
bunch_position_gauss, bunch_length_gauss, fitparams_gauss = gaussianFit(
    time_array, gaussian_profile, fitOpt=fitOpt)

fitOpt = FitOptions(fittingRoutine='minimize')
bunch_position_binom, bunch_length_binom, fitparams_binom = binomialAmplitudeNFit(
    time_array, binom_profile, fitOpt=fitOpt)

print('Gauss: Initial ->',initial_params_gauss, '/ Final ->', fitparams_gauss)
print('Binomial: Initial ->',initial_params_binom, '/ Final ->', fitparams_binom)


# In[15]:


# Using custom FitOptions
# NB: a new FitOptions should be created to reset initial conditions,
# but the same FitOptions can be used to share the same initial conditions

from blond_common.fitting.distribution import gaussianFit, parabolicAmplitudeFit, binomialAmplitudeNFit, FitOptions

fitOpt = FitOptions(fittingRoutine='minimize',
                    method=None, # method='Nelder-Mead' or method='Powell' or method=None
                    options={'disp':True})

bunch_position_binom, bunch_length_binom, fitparams_binom = binomialAmplitudeNFit(
    time_array, binom_profile, fitOpt=fitOpt)

print('Binomial: Initial ->',initial_params_binom, '/ Final ->', fitparams_binom)


# In[16]:
# Compute the spectra of the distributions and compare with numerical Fourier
# transform

frequency_array = np.linspace(0, 500e6)

gauss_spectrum = gaussian_dist.spectrum(frequency_array)
binom_spectrum = binom_dist.spectrum(frequency_array)

dt = time_array[1] - time_array[0]
gauss_spectrum_DFT = np.zeros(len(frequency_array), dtype=complex)
binom_spectrum_DFT = np.zeros_like(gauss_spectrum_DFT)
for it, f in enumerate(frequency_array):
    gauss_spectrum_DFT[it] = np.trapz(gaussian_profile*np.exp(-2j*np.pi*f*time_array), dx=dt)
    binom_spectrum_DFT[it] = np.trapz(binom_profile*np.exp(-2j*np.pi*f*time_array), dx=dt)

plt.figure('gauss spectrum', clear=True)
plt.grid()
plt.plot(frequency_array, gauss_spectrum.real, label='analytic, real')
plt.plot(frequency_array, gauss_spectrum_DFT.real, 'C0.', label='numeric, real')
plt.plot(frequency_array, gauss_spectrum.imag, label='analytic, imag')
plt.plot(frequency_array, gauss_spectrum_DFT.imag, 'C1.', label='numeric, real')
plt.legend()

plt.figure('binomial spectrum', clear=True)
plt.grid()
plt.plot(frequency_array, binom_spectrum.real, label='analytic, real')
plt.plot(frequency_array, binom_spectrum_DFT.real, 'C0.', label='numeric, real')
plt.plot(frequency_array, binom_spectrum.imag, label='analytic, imag')
plt.plot(frequency_array, binom_spectrum_DFT.imag, 'C1.', label='numeric, imag')
plt.legend()
