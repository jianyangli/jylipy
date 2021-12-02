import numpy as np
from ..vector import *
from ..saoimage import getds9

class TestEllipsoidProjection():
    def test_xy2lonlat_sphere(self):
        y, x = np.mgrid[0:511:512j, 0:511:512j]
        los_lon, los_lat, pa = np.random.rand(3)
        los_lon = los_lon * 360
        los_lat = los_lat * 180 - 90
        pa = pa * 180
        center = []
        los = Vector(1, los_lon, los_lat, type='geo', deg=True)
        r = 300
        pxlscl = 2
        center = np.random.rand(2) * 10 + 250.5
        v = EllipsoidProjection(r, los, pxlscl, pa=pa, center=center)
        lon, lat = v.xy2lonlat(x, y)
        x1, y1 = v.lonlat2xy(lon, lat)
        ww = np.isfinite(x) & np.isfinite(x1)
        assert np.allclose(x[ww], x1[ww])
        assert np.allclose(y[ww], y1[ww])

    def test_lonlat2xy_sphere(self):
        lat, lon = np.mgrid[-90:90:181j, 0:359:360j]
        los_lon, los_lat, pa = np.random.rand(3)
        los_lon = los_lon * 360
        los_lat = los_lat * 180 - 90
        pa = pa * 180
        los = Vector(1, los_lon, los_lat, type='geo', deg=True)
        r = 300
        pxlscl = 2
        center = np.random.rand(2) * 10 + 250.5
        v = EllipsoidProjection(r, los, pxlscl, pa=pa, center=center)
        x, y = v.lonlat2xy(lon, lat)
        lon1, lat1 = v.xy2lonlat(x, y)
        ww = np.isfinite(lon) & np.isfinite(lon1) & (abs(lat) != 90)
        diff = (lon[ww] - lon1[ww]) % 360
        diff[diff > 180] -= 360
        assert np.allclose(diff, 0)
        diff = (lat[ww] - lat1[ww]) % 360
        diff[diff > 180] -= 360
        assert np.allclose(diff, 0)

    def test_xy2lonlat_ellipsoid(self):
        y, x = np.mgrid[0:511:512j, 0:511:512j]
        lon, lat, pa = np.random.rand(3)
        lon = lon * 360
        lat = lat * 180 - 90
        pa = pa * 180
        los = Vector(1, lon, lat, type='geo', deg=True)
        r = [300, 250, 180]
        pxlscl = 2
        center = np.random.rand(2) * 10 + 250.5
        v = EllipsoidProjection(r, los, pxlscl, pa=pa, center=center)
        lon, lat = v.xy2lonlat(x, y)
        x1, y1 = v.lonlat2xy(lon, lat)
        ww = np.isfinite(x) & np.isfinite(x1)
        assert np.allclose(x[ww], x1[ww])
        assert np.allclose(y[ww], y1[ww])

    def test_lonlat2xy_ellipsoid(self):
        lat, lon = np.mgrid[-90:90:181j, 0:359:360j]
        los_lon, los_lat, pa = np.random.rand(3)
        los_lon = los_lon * 360
        los_lat = los_lat * 180 - 90
        pa = pa * 180
        los = Vector(1, los_lon, los_lat, type='geo', deg=True)
        r = [300, 250, 180]
        pxlscl = 2
        center = np.random.rand(2) * 10 + 250.5
        v = EllipsoidProjection(r, los, pxlscl, pa=pa, center=center)
        x, y = v.lonlat2xy(lon, lat)
        lon1, lat1 = v.xy2lonlat(x, y)
        ww = np.isfinite(lon) & np.isfinite(lon1) & (abs(lat) != 90)
        diff = (lon[ww] - lon1[ww]) % 360
        diff[diff > 180] -= 360
        assert np.allclose(diff, 0)
        diff = (lat[ww] - lat1[ww]) % 360
        diff[diff > 180] -= 360
        assert np.allclose(diff, 0)
