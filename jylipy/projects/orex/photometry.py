# photometric models for OSIRIS-REx

import numpy as np
from astropy.modeling import Parameter
from astropy.io import fits
import jylipy.Photometry as pho


def _2rad(*args):
    return tuple(map(np.deg2rad, args))


# default unit for angles is degrees

def expoly(alpha, b, c, d):
    """Exponential polynomial model"""
    alpha2 = alpha*alpha
    return np.exp(b*alpha + c*alpha2 + d*alpha*alpha2)


def ls_disk(inc, emi):
    inc, emi = _2rad(inc, emi)
    return np.cos(inc) / (np.cos(inc) + np.cos(emi))


class LS(pho.PhotometricModel):
    """Lommel-Seeliger model"""
    A0 = Parameter()
    b = Parameter()
    c = Parameter()
    d = Parameter()

    @staticmethod
    def evaluate(inc, emi, pha, A0, b, c, d):
        disk = ls_disk(inc, emi)
        f = expoly(pha, b, c, d)
        return A0 * f * disk


class ROLO(pho.PhotometricModel):
    """ROLO model"""
    C0 = Parameter()
    C1 = Parameter()
    A0 = Parameter()
    A1 = Parameter()
    A2 = Parameter()
    A3 = Parameter()
    A4 = Parameter()

    @staticmethod
    def evaluate(inc, emi, pha, C0, C1, A0, A1, A2, A3, A4):
        opsur = C0 * np.exp(-C1 * pha)
        alpha2 = pha * pha
        f = opsur + A0 + A1*pha + A2*alpha2 + A3*pha*alpha2 + A4*alpha2*alpha2
        d = ls_disk(inc, emi)
        return f * d / np.pi


class Minnaert(pho.PhotometricModel):
    """Minnaert model"""
    Am = Parameter()
    b = Parameter()
    c = Parameter()
    d = Parameter()
    k0 = Parameter()
    b1 = Parameter()

    @staticmethod
    def evaluate(inc, emi, pha, Am, b, c, d, k0, b1):
        pha2 = pha * pha
        f = 10**(-0.4 * (b*pha + c*pha2 + d*pha*pha2))
        k = k0 + b1*pha
        inc, emi = _2rad(inc, emi)
        mu0 = np.cos(inc)
        mu = np.cos(emi)
        disk = (mu0 * mu)**(k-1) * mu0
        return Am * f * disk


class McEwen(pho.PhotometricModel):
    """McEwen model"""
    Amc = Parameter()
    b = Parameter()
    c = Parameter()
    d = Parameter()
    e = Parameter()
    f = Parameter()
    g = Parameter()

    @staticmethod
    def evaluate(inc, emi, pha, Amc, b, c, d, e, f, g):
        f = expoly(pha, e, f, g)
        L = expoly(pha, b, c, d)
        disk = ls_disk(inc, emi)
        inc = _2rad(inc)
        return Amc * f * (2*L*disk + (1-L)*np.cos(inc))


class Akimov(pho.PhotometricModel):
    """Akimov model"""

    inputs = ('pha', 'lat', 'lon')

    Aak = Parameter()
    b = Parameter()
    c = Parameter()
    d = Parameter()

    @staticmethod
    def evaluate(pha, lat, lon, Aak, b, c, d):
        f = expoly(pha, b, c, d)
        disk = pho.AkimovDisk(Aak)
        return np.pi * f * disk(pha, lat, lon)


class Akimov_Linear(pho.PhotometricModel):
    """Akimov-linear model"""

    inputs = ('pha', 'lat', 'lon')

    ALak = Parameter()
    b = Parameter()

    @staticmethod
    def evaluate(pha, lat, lon, ALak, b):
        f = 10**(0.4*b*pha)
        disk = pho.AkimovDisk(ALak)
        return np.pi * f * disk(pha, lat, lon)


class PhotometricData(pho.PhotometricData):

    @classmethod
    def from_spdif(cls, spdif):
        """Initialize PhotometricData class from SPDIF"""

        data = fits.open(spdif)
        pho = cls(iof=data[0].data,
                  inc=data[3].data['incidang'], emi=data[3].data['emissang'],
                  pha=data[3].data['phaseang'], geolon=data[3].data['lon'],
                  geolat=data[3].data['lat'], band=data['wavelength'].data)

        return pho


class GetSpots(dict):
    """Class to store the SCLK list for all facets"""

    @classmethod
    def from_getspots(cls, facetfile):
        """Initialize class object from input getspots file"""
        obj = cls()
        # Read facet file
        with open(facetfile) as f:
            lines = f.read().strip().split('\n')
        sclks = []
        facet = lines[0].strip()
        for l in lines[1:]:
            if l.startswith('F') or l.startswith('END'):
                if len(sclks) > 0 :
                    # After F1, do the save to filename then reset sclks
                    obj[facet] = sclks.copy()
                sclks = []
                facet = l.strip()
            else:
                sclks.append(l.split()[0])
        return obj

    @property
    def n_spots(self):
        return [len(self[k]) for k in self]

    def extract_phodata(self, spdif, outfile='phoarray.fits'):
        """Extract photometric data from SPDIF file

        Parameters
        ----------
        spdif : str
            File name of the input SPDIF
        outfile : str, optional
            The name of output file.

        Returns
        -------
        `PhotometricDataArray`
            The photometric data array extracted from SPDIF based on
            `GetSpots` result
        """
        nfac = len(self)
        print(f'Data from {nfac} facets to be extracted.')
        indata = fits.open(spdif)
        sclks = list(indata[3].data['SCLK'])
        keys = list(self.keys())
        outdata = pho.PhotometricDataArray(nfac)
        for i in range(nfac):
            print('                                                          '
                  '                                                          '
                  '                                          ', end='\r')
            print(f'Processing facet {keys[i]}, {len(self[keys[i]])} SCLK to'
                   ' be checked.  ', end=' ')
            w = []
            for s in self[keys[i]]:
                if s in sclks:
                    w.append(sclks.index(s))
            if len(w) > 0:
                outdata[i] = PhotometricData(inc=indata[3].data['incidang'][w],
                                             emi=indata[3].data['emissang'][w],
                                             pha=indata[3].data['phaseang'][w],
                                             iof=indata[0].data[w],
                                             geolat=indata[3].data['lat'][w],
                                             geolon=indata[3].data['lon'][w],
                                             band=indata[2].data)
            #print(f'{len(outdata[i])} spots collected.', end='\r')
        return outdata
