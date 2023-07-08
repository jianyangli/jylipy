import numpy as np
import matplotlib.pyplot as plt
from ..models import STM, NEATM
from ..vector import Vector


def test_transformation():
    stm = STM(3 * u.au, 10 * u.km)
    y, z = np.meshgrid(range(-50, 51), range(-50, 51))
    x = np.sqrt(50**2 - y**2 - z**2)
    inside = np.isfinite(x)
    vv = Vector(x[inside], y[inside], z[inside])
    m = stm._transfer_to_bodyframe(-45*u.deg, -45*u.deg)
    intfunc = np.full_like(x, np.nan)
    ff = [stm._int_func(x, y, m, 'W m-2 Hz-1 sr-1', 3*u.um) for x, y in zip(vv.lon, vv.lat)]

    intfunc[inside] = u.Quantity(ff).value
    plt.imshow(intfunc)
