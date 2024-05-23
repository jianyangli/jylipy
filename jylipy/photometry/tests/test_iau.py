"""Test IAU models"""


import numpy as np
import astropy.units as u
from jylipy.photometry.iau import *
from jylipy.photometry.hapke import DiskInt5


def test_default_keyword():

    @default_keyword(test='test')
    def test(**kwargs):
        return kwargs

    kwargs = test()
    assert 'test' in kwargs
    assert kwargs['test'] == 'test'
    kwargs = test(test='already set')
    assert kwargs['test'] == 'already set'


class testHG():

    def test__init__(self):
        m = HG(H=3 * u.mag, G=0.12, radius=500 * u.km)
        assert m.wfb == 'V'
        assert u.isclose(m.geomalb, 0.11063966)
        assert u.isclose(m.phaseint, 0.3643505755292939)
        assert u.isclose(m.bondalb, 0.04031162)

    def test_toHapke(self):
        m = HG(H=3 * u.mag, G=0.12, radius=500 * u.km)
        hapke = m.toHapke()
        assert isinstance(hapke, DiskInt5)
        assert np.isclose(hapke.w.value, 0.11642664)
        assert np.isclose(hapke.g.value, -0.27175238)
        assert np.isclose(hapke.theta.value, 0.)
        assert np.isclose(hapke.B0.value, 2.16975104)
        assert np.isclose(hapke.h.value, 0.0509448)


class testHG1G2():

    def test__init__(self):
        m = HG1G2(H=3 * u.mag, G1=0.12, G2=0.4, radius=500 * u.km)
        assert m.wfb == 'V'
        assert u.isclose(m.geomalb, 0.11063966)
        assert u.isclose(m.phaseint, 0.381494)
        assert u.isclose(m.bondalb, 0.04220836)


class testHG12():

    def test__init__(self):
        m = HG12(H=3 * u.mag, G12=0.4, radius=500 * u.km)
        assert m.wfb == 'V'
        assert u.isclose(m.geomalb, 0.11063966)
        assert u.isclose(m.phaseint, 0.40582662035560013)
        assert u.isclose(m.bondalb, 0.04490052)
