import numpy as np
import os, astropy.units as u
import matplotlib.dates as mdates
from astropy.io import ascii, fits
from astropy.time import Time
from astropy import table
from photutils import CircularAperture, aperture_photometry
from sbpy.units import VEGAmag, VEGA
from jylipy.hst.wfc3 import UVISImage, PAM, uvis_pix


u.add_enabled_units(VEGAmag)
u.add_enabled_units(VEGA)


visits = ['0o', '01', '02', '03', '04', '05', '06', '11', '12', '13',
          '14', '15', '16', '17', '18', '21', '22', '23', '24']


# uvis2 photcal constants from
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
phocal = {'filter': 'F350LP',
          'photplam': 5851.14829 * u.AA,
          'photbw': 1483.01704 * u.AA,
          'abmag': 26.936 * u.mag,
          'vegamag': 26.78 * VEGAmag,
          'stmag': 27.08 * u.mag,
          'error': 0.005 * u.mag,
          'photflam': 5.3469E-20 * u.Unit('erg cm-2 AA-1 electron-1'),
          'err_photflam': 2.3475E-22 * u.Unit('erg cm-2 AA-1 electron-1')
         }
phocal['fluxzpt'] = np.round((2.5 * np.log10(phocal['photflam'].to_value(
            'J / (m2 um electron)')) + phocal['vegamag'].value) * VEGAmag, 2)


class PhotometryError(Exception):
    pass


class Photometry():
    """Didymos-Dimorphos photometry
    
    Attributs
    ---------
    datapath : str
        Path to data
    info : astropy.table.Table
        Information table.
    aperture : number array
        Aperture radii.  Added by `.measure()`
    counts : Quantity
        Aperture photometry in e-.  Added by `.measure()`
    ct_err : Quantity
        Aperture photometry error in e-.  Added by `.measure()`, will
        be modified by `photometric_cal()` for calibration.
    flux, flux_err : Quantity
        Aperture integrated flux and error.  Added by `.photometric_cal()`.
    mag, mag_err : Quantity
        Integrated magnitude and error.  Added by `.photometric_cal()`.
    """
    def __init__(self, visits=visits,
                 datapath=os.path.join('..', 'data'),
                 ctfile='meta/centroid_{}.csv',
                 bgfile='meta/bg_{}.csv',
                 aspfile='meta/aspect_{}.ecsv'):

        self.datapath = datapath
        info = []
        for v in visits:
            asp = ascii.read(os.path.join(datapath, aspfile.format(v)))
            bg = ascii.read(os.path.join(datapath, bgfile.format(v)))
            if 'ext' in bg.colnames:
                bg.remove_column('ext')
            ct = ascii.read(os.path.join(datapath, ctfile.format(v)))
            for k in ['ext', 'status']:
                if k in ct.colnames:
                    ct.remove_column(k)
            
            info.append(table.join(table.join(asp, bg, keys='file'), ct,
                                   keys='file'))

        info = table.vstack(info)
        info.sort('utc-mid')
        self.info = info
        self.fields = []
    
    @u.quantity_input(aperture=[u.pix, u.km])
    def measure(self, aperture=np.linspace(1, 130, 130) * u.pix):
        """Measure aperture photometry
        
        info : astropy.table.Table
            Information table that lists the information of all images
            for photometric measurements
        aperture : array
            Aperture sizes
            
        Write photometry to self.counts
        Write error to self.ct_err
        """
        self.aperture = aperture
        apt = aperture
        pam = PAM()  # pixel area map
        phos = []
        errs = []
        # maximum possible aperture radius, determined by the position of target
        max_r = []
        for r in self.info:

            im = fits.open(os.path.join(self.datapath,
                                        r['file'].replace('_flt', '_flc')))
            # calculate maximum aperture radius that is fully inside the image
            max_r.append(np.min([abs(r['yc'] - im['sci'].data.shape[0]),
                                 abs(r['xc'] - im['sci'].data.shape[1])]))
            # pam correction
            pam_arr = pam(r['aperture'])
            if '512' in r['aperture']:
                # pad 512x512 pam to 512x513 to match data, no idea why
                pam_arr = np.pad(pam_arr, [[0, 0], [0, 1]])
            data = im['sci'].data * pam_arr
            err = im['err'].data * pam_arr
            # process aperture size
            if aperture.unit.is_equivalent('km'):
                apt = (aperture / r['range']).to_value('arcsec',
                            u.dimensionless_angles()) / 0.04
            else:
                apt = aperture.value
            # measure photometry
            pho = table.vstack([aperture_photometry(data,
                        CircularAperture([r['xc'], r['yc']], i)) for i in apt])
            pho = pho['aperture_sum']
            pho_arr = np.asarray(pho).view(float).reshape(-1)
            phos.append(pho_arr)
            # estimate error
            pho = table.vstack([aperture_photometry(data,
                        CircularAperture([r['xc'], r['yc']], i), error=err,
                                         mask=im['dq'].data>0) for i in apt])
            err = pho['aperture_sum_err']
            err_arr = np.asarray(err).view(float).reshape(-1)
            errs.append(err_arr)
        
        # save results
        self.info.add_column(table.Column(max_r, name='max_apt'))
        self.counts = u.Quantity(phos, u.electron).T
        self.ct_err = u.Quantity(errs, u.electron).T
        self.fields.extend(['counts', 'ct_err'])
    
    def photometric_cal(self):
        """Photometric calibration
        """
        # background removal
        self.counts -= u.Quantity(np.outer(np.pi * self.aperture.value**2,
                                  self.info['background']), u.electron)
        self.ct_err =  np.sqrt(self.ct_err**2 + u.Quantity(np.outer(
            np.pi * self.aperture.value**2, self.info['background_error']),
            u.electron)**2)
        
        # normalize exposure time
        self.counts /= u.Quantity(self.info['exptime'], u.s)
        self.ct_err /= u.Quantity(self.info['exptime'], u.s)
                
        # convert to flux
        self.flux = (self.counts * phocal['photflam']).to('W / m2 um')
        self.flux_err = (self.ct_err * phocal['photflam']).to('W / m2 um')
        
        # convert to vega mag
        self.mag = u.Quantity(-2.5 * np.log10(self.counts.value), u.mag) \
                   + phocal['vegamag']
        self.mag_err =u.Quantity( -2.5 * np.log10(self.ct_err / self.counts
                        + 1).value, u.mag)

        # aperture correction for pre-impact point source
        # nn = 4  # 4 pre-impact images
        ee_corr = ee(self.aperture.to('arcsec', uvis_pix()))
        self.counts_apc = (self.counts.T / ee_corr).T
        self.ctapc_err = (self.ct_err.T / ee_corr).T

        # convert to flux
        self.flux_apc = (self.counts_apc * phocal['photflam']).to('W / m2 um')
        self.fluxapc_err = (self.ctapc_err * phocal['photflam']).to('W / m2 um')
        
        # convert to vega mag
        self.mag_apc = u.Quantity(-2.5 * np.log10(self.counts_apc.value),
                    u.mag) + phocal['vegamag']
        self.magapc_err =u.Quantity( -2.5 * np.log10(self.ctapc_err /
                    self.counts + 1).value, u.mag)
        
        self.fields.extend(['flux', 'flux_err', 'mag', 'mag_err',
                            'counts_apc', 'ctapc_err', 'flux_apc',
                            'fluxapc_err', 'mag_apc', 'magapc_err'])

    def write(self, outfile, overwrite=False):
        """Save photometry to fits file"""
        hdu = fits.PrimaryHDU()
        hdu.header['datapath'] = self.datapath
        hdulist = fits.HDUList([hdu])
        for k in self.fields + ['aperture']:
            self._add_attr(k, hdulist)
        hdu_tbl = fits.BinTableHDU(self.info, name='info')
        hdulist.append(hdu_tbl)
        hdulist.writeto(outfile, overwrite=overwrite)

    def _add_attr(self, attr, hdulist):
        """Add attribution 'attr' to fits.HDUList.
        """
        if hasattr(self, attr):
            att = getattr(self, attr)
            hdu = fits.ImageHDU(getattr(att, 'value', att).astype('float32'),
                                name=attr)
            hdu.header['bunit'] = str(getattr(att, 'unit', ''))
            hdulist.append(hdu)

    def merge(self, pho):
        ### TO BE COMPLETED
        """Merge two Photometry objects
        
        Both objects must have the same `.fields`
        
        pho : Photometry
            The object to be merged.
        """
        if not isinstance(pho, Photometry):
            raise PhotometryError('A `Photometry` object is required, '
                '{} received.'.format(type(pho)))
        if not np.allclose(self.aperture, pho.aperture):
            raise PhotometryError('Input `Photometry` object cannot be '
                'merged: Different aperture sizes.')
        self.info = table.join(self.info, pho.info, join_type='left')
        self.info.sort('utc-mid')
        for k in self.fields:
            setattr(self, k, np.concatenate([getattr(self, k),
                    getattr(pho, k)], axis=1))
            
    @classmethod
    def read(cls, infile):
        obj = cls()
        obj.fields = []
        with fits.open(infile) as f_:
            obj.datapath = f_[0].header['datapath']
            for n in range(len(f_) - 1):
                value = f_[n+1].data
                if 'bunit' in f_[n+1].header:
                    value = value * u.Unit(f_[n+1].header['bunit'])
                fld = f_[n+1].header['extname'].lower()
                setattr(obj, fld, value)
                obj.fields.append(fld)
        obj.fields.remove('info')
        obj.info = table.Table(obj.info)
        return obj
            
    def plot(self, aperture=None, apc=True, figsize=(6, 4)):
        # plot aperture photometry
        if aperture is None:
            plot_apt = self.aperture
        else:
            plot_apt = aperture
        
        short = self.info['exptime'] < 20
        tt = Time(self.info['utc-mid'][short])
        if apc:
            mag_plot = self.mag_apc[:, short]
            mag_err = self.magapc_err[:, short]
        else:
            mag_plot = self.mag[:, short]
            mag_err = self.mag_err[:, short]
        
        f, ax = plt.subplots(figsize=figsize)
        for i in np.asarray(plot_apt, dtype=int) - 1:
            #plt.plot_date(tt.plot_date, mag_plot[i], 'o', mfc='none')
            ax.errorbar(tt.plot_date, mag_plot[i].value, mag_err[i].value,
                fmt='o', mfc='none')
        ax.legend(['{}'.format(x) for x in (plot_apt * u.pix).to('arcsec',
                uvis_pix())], ncol=2, loc='upper right')
        jp.pplot(ax, ylabel=self.mag.unit, skipline=True, ylim=[15, 13.2])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%dT%H:%M'))
        _ = plt.xticks(rotation=45)
        ax.grid()
        
        return ax
    
    def to_table(self, fields=None):
        """Export photometry to astropy.table.Table
        
        fields : array of str
            The fields to be included in the table.  Default is all fields:
            ['counts', 'ct_err', 'counts_apc', 'ctapc_err', 'flux',
             'flux_err', 'mag', 'mag_err']
        """
        out = self.info.copy()
        if fields is None:
            fields = self.fields
        for k in fields:
            if hasattr(self, k):
                v = getattr(self, k)
                for i, a in enumerate(self.aperture):
                    colname = '{}_{}{}'.format(k, a.value, a.unit)
                    out.add_column(table.Column(v[i], name=colname))
        return out
    
