# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for distribution.py
:Authors: **Markus Schwarz**
"""

# General imports
# ---------------
import sys
import unittest
import numpy as np

# BLonD imports
# -------------
sys.path.append('./../../../')
from blond_common.fitting import distributions


class TestDistributionsBaseClass(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        self.test_object = distributions._DistributionObject()

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """

        standardMsg = "%s is not NaN" % str(value)

        if not np.isnan(value):
            self.fail(self._formatMessage(msg, standardMsg))
            
    def assertHasAttribute(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)

        self.assertTrue(testBool, msg='obj lacking an attribute. obj: %s, intendedAttr: %s' % (obj, intendedAttr))

    ### test helper function
    def test_check_greater_zero(self):
        # Test if ValueError is raised for negative arguments
        with self.assertRaises(ValueError):
            distributions.check_greater_zero(-42, '')
    
    ### test base class
    def test_base_class_attribute_amplitude(self):
        # Test if base class has amplitude
        self.assertHasAttribute(self.test_object, 'amplitude')
    
    def test_base_class_attribute_position(self):
        # Test if base class has amplitude
        self.assertHasAttribute(self.test_object, 'position')
        
    def test_base_class_attribute_RMS(self):
        # Test if base class RMS returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.RMS
    
    def test_base_class_attribute_FWHM(self):
        # Test if base class FWHM returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.FWHM
    
    def test_base_class_attribute_fourSigma_RMS(self):
        # Test if base class fourSigma_RMS returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.fourSigma_RMS
    
    def test_base_class_attribute_fourSigma_FWHM(self):
        # Test if base class fourSigma_FWHM returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.fourSigma_FWHM

    def test_base_class_attribute_full_bunch_length(self):
        # Test if base class full_bunch_length returns RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.full_bunch_length

    def test_base_class_profile(self):
        # Test if base class.profile() throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.profile()

    def test_base_class_attribute_spectrum(self):
        # Test if base class.spectrum() throws RuntimeError
        with self.assertRaises(RuntimeError):
            self.test_object.profile()


class TestDistributionsGaussianClass(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):
        self.gaussian_object = distributions.Gaussian(1, 0.0, 1)
    
    def test_negative_bunch_length_exception(self):
        with self.assertRaises(ValueError):
            distributions.Gaussian(1, 0.0, -42)
    
    def test_amplitude(self):
        self.assertEqual(self.gaussian_object.amplitude, 1)
    
    def test_position(self):
        self.assertEqual(self.gaussian_object.position, 0.0)
    
    def test_RMS(self):
        self.assertEqual(self.gaussian_object.RMS, 1)
    
    def test_FWHM(self):
        self.assertEqual(self.gaussian_object.FWHM, 2*np.sqrt(np.log(4))*1)

    def test_RMS_update(self):
        # test if RMS updates correctly when FWHM is changed
        new_FWHM = 42
        self.gaussian_object.FWHM = new_FWHM
        self.assertEqual(self.gaussian_object.RMS, 17.8357578060484)

if __name__ == '__main__':

    unittest.main()
