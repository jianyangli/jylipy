"""Expanding xarray.DataArray

Add functionality to save and load `xarray.DataArray` to and from FITS file

"""

import xarray as xr
from astropy.io import fits
from astropy.table import Table


__all__ = ['DataArray', 'open_dataarray']


class DataArray(xr.DataArray):

    def to_fits(self, file, **kwargs):
        """Save to FITS file

        FITS data structure:
        - Primary HDU : Empty data.  Headers saves the information about this
          DataArray.
          Header keyword:
            NAME : str, name of data array or data set
            NARRAY : num, number of data arrays contained in the file
            NCOORDS : int, number of coordinates
            CNAME# : str, name of the #th coordinate
        - 1st HDU : DataArray.data
            NAXIS : FITS standard keyword, int, number of dimensions (axes)
            DNAME# : str, name of the #th dimension
            NAME : str, data array name
        - 2nd HDU : DataArray.attrs in binary table, or empty
        - 3rd HDU - Nth HDU : DataArray.coords
            NAXIS : FITS standard keyword, int, number of dimensions (axes)
            DNAME# : str, name of the #th dimension
            NAME : str, coordinate array name
        """
        # primary extension
        hdu = fits.PrimaryHDU()
        hdu.header['name'] = self.name, 'DataArray name'
        hdu.header['narray'] = 1, 'Number of data arrays in the fits'
        hdu.header['ncoords'] = len(self.coords), 'Number of coordinates'
        for i, cn in enumerate(self.coords.keys()):
            hdu.header['cname{}'.format(i)] = cn, 'Coordinate {} name'
        hdulist = fits.HDUList([hdu])
        # first extension
        hdu = fits.ImageHDU(self.data)
        hdu.header['name'] = self.name, 'DataArray name'
        for i, dn in enumerate(self.dims):
            hdu.header['dname{}'.format(i)] = dn, \
                                              'Dimension {} name'.format(i)
        hdulist.append(hdu)
        # second extension, attributs
        attrs = self.attrs.copy()
        for k, v in attrs.items():
            if isinstance(v, str) or (not hasattr(v, '__iter__')):
                attrs[k] = [v]
        attr_table = Table(attrs)
        attr_hdu = fits.BinTableHDU(attr_table)
        hdulist.append(attr_hdu)
        # coordinate extensions
        for i, c in enumerate(self.coords.keys()):
            hdu = fits.ImageHDU(self.coords[c])
            for i, dn in enumerate(self.coords[c].dims):
                hdu.header['dname{}'.format(i)] = dn, \
                                                  'Dimension {} name'.format(i)
            hdu.header['name'] = self.coords[c].name, \
                                 'Coordinate {} name'.format(i)
            hdulist.append(hdu)
        hdulist.writeto(file, **kwargs)


def open_dataarray(file):
    pass
    if not isinstance(file, (str, fits.HDUList)):
        pass

