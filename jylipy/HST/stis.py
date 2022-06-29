"""HST STIS image related functionalities"""

__all__ = ['STISImage']

import ccdproc
from astropy import nddata
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from ..core import Image

class STISImage(Image):

    @classmethod
    def from_fits(cls, file):
        with fits.open(file) as f_:
            hdr0 = f_[0].header
            sci = u.Quantity(f_[1].data, 'count')
            hdr1 = f_[1].header
            err = nddata.StdDevUncertainty(f_[2].data, unit=u.ct)
            hdr2 = f_[2].header
            dq = f_[3].data
            hdr3 = f_[3].header
        mask = dq != 0
        return cls(data=sci, uncertainty=err, mask=mask, wcs=WCS(hdr1),
                   meta={'header0': hdr0,
                         'header1': hdr1,
                         'header2': hdr2,
                         'header3': hdr3})
