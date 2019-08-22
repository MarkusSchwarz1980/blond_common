# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:17:25 2019

@author: schwarz
"""

class ValidateInStrings(object):
    def __init__(self, key, valid, ignorecase=False):
        'valid is a list of legal strings'
        self.key = key
        self.ignorecase = ignorecase

        def func(s):
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}

    def __call__(self, s):
        if self.ignorecase:
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        raise ValueError('Unrecognized %s string %r: valid strings are %s'
                         % (self.key, s, list(self.valid.values())))

def validate_scale_factor(bl):
    if isinstance(bl, str):
        return _validate_named_scale_factor(bl)


def validate_float(s):
    """Convert s to float or raise."""
    try:
        return float(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to float' % s)

_validate_named_scale_factor = ValidateInStrings('scale_factor',
         ['RMS','FWHM','fourSigma_RMS','fourSigma_FWHM','full_bunch_length'])

_defaultBLonDRcParams = {
        'distribution.scale_factor' : ['RMS', validate_scale_factor]
        }