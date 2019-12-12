# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for computing induced voltage
:Authors: **Simon Albright**
"""

#General imports
import numpy as np


#BLonD_Common imports
if __name__ != '__main__':
    from . import impedance_sources as impSource
else:
    import blond_common.interfaces.impedances.impedance_sources as impSource


class InducedVoltage:
    
    def __init__(self, impedance_list = [], wake_list = []):
        
        self.impedances_loaded = impedance_list
        self.wakes_loaded = wake_list
        
        self.interp_frequency_array = None
        self.interp_time_array = None


    def sum_impedance_sources(self):
        
        if len(self.impedances_loaded) > 0:
            self.total_impedance = np.zeros(len(self.interp_frequency_array), \
                                            dtype='complex')
            for imp in self.impedances_loaded:
                imp.imped_calc(self.interp_frequency_array)
                self.total_impedance += imp.impedance
                
        if len(self.wakes_loaded) > 0:
            self.total_wake = np.zeros(len(self.interp_time_array))
            for wake in self.wakes_loaded:
                wake.wake_calc(self.interp_time_array)
                self.total_wake += wake.wake


    def _calc_induced_time(self):
        
        calc_induced_time(self.profile, se)

                
def calc_induced_freq(self, spectrum, impedance):
    
    return spectrum*impedance

def calc_induced_time(self, profile, wake):
    
    return np.convolve(profile, wake)
        
        




if __name__ == '__main__':
    
    InducedVoltage()