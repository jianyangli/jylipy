from spiceypy import *

@spiceErrorCheck
def kepleq(ml, h, k):
    """
    Return the state (position and velocity) of a target body
    relative to an observing body, optionally corrected for light
    time (planetary aberration) and stellar aberration.

    http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html

    :param targ: Target body name.
    :type targ: str
    :param et: Observer epoch.
    :type et: float
    :param ref: Reference frame of output state vector.
    :type ref: str
    :param abcorr: Aberration correction flag.
    :type abcorr: str
    :param obs: Observing body name.
    :type obs: str
    :return:
            State of target,
            One way light time between observer and target.
    :rtype: tuple
    """
    ml = ctypes.c_double(ml)
    h = ctypes.c_double(h)
    k = ctypes.c_double(k)
    f = ctypes.c_double()
    libspice.kepleq_c(ml, h, k, ctypes.byref(f))
    return f.value

