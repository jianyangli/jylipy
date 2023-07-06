import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc as ccdp

class CRRej():
    """Cosmic ray rejection for HST images.  This class only works
    for '_flt' or '_flc' images.
    """
    
    def __init__(self, sigclip=4, cleantype='medmask', niter=10,
                 masks=[4, 256, 512], untouch=0):
        """
        sigclip, cleantype, niter
            Parameters for `ccdproc.cosmicray_lacosmic`
        mask : list of int, optional
            The UVIS DQ mask values to be used for the cosmic ray
            rejection
        untouch : int, optional
            The side of the center box that is untouched by cosmic ray
            rejection process.
        """
        self.sigclip = sigclip
        self.cleantype = cleantype
        self.niter = niter
        self.masks = masks
        self.untouch = untouch

    def clean(self, infile, centroid=None, verbose=True):
        """
        infile : str
            Name of input fits file
        centroid : np.array of shape (2,), optional
            Specify the centroid (y, x).  Used when `.untouch > 0`.
            If `.untouch > 0` but `centroid == None`, then a
            ValueError will be raised.
        verbose : bool, optional
            Verbose mode
        """
        # load image
        im = CCDData.read(infile)
        with fits.open(infile) as f_:
            dq = f_['dq'].data
            rn = f_[0].header['PCTERNOI']
        if self.untouch > 0:
            if centroid is None:
                raise ValueError('Centroid is not specified')
            halfside = self.untouch // 2
            y1 = int(centroid[0]) - halfside
            y2 = int(centroid[0]) + halfside + 1
            x1 = int(centroid[1]) - halfside
            x2 = int(centroid[1]) + halfside + 1
            box = im.data[y1:y2, x1:x2]
        
        # generate mask from dq array
        mask = np.zeros_like(dq)
        for v in self.masks:
            mask |= dq & v
        im.mask = mask > 0

        # remove cosmic ray
        self.cleaned = ccdp.cosmicray_lacosmic(im, readnoise=rn,
            sigclip=self.sigclip, cleantype=self.cleantype, niter=self.niter,
            verbose=verbose)
        # save cosmic ray mask
        crmask = self.cleaned.mask
        crmask[mask > 0] = False
        self.cleaned.crmask = crmask

        # recover the center region
        if self.untouch > 0:
            #return box, y1, y2, x1, x2
            self.cleaned.data[y1:y2, x1:x2] = box
            self.cleaned.crmask[y1:y2, x1:x2] = False

    def write(self, outfile, infile=None, overwrite=False):
        """
        If input file is provided, then the 'sci' and 'dq' segments will
        be updated with the cleaned image and cosmic ray mask.  Otherwise
        a new HDU list will be generated that contains the 'sci' and 'dq'
        segments.
        
        The 'dq' mask marks the cosmic ray affected pixels by 4096.
        """
        if infile is not None:
            f_ = fits.open(infile)
        else:
            f_ = fits.HDUList([fits.PrimaryHDU()])
            f_.append(fits.ImageHDU(np.zeros_like(self.cleaned.data,
                        dtype='float32'), name='sci'))
            f_.append(fits.ImageHDU(np.zeros_like(self.cleaned.data,
                        dtype='int16'), name='dq'))
        f_['sci'].data = self.cleaned.data.astype('float32')
        f_['dq'].data = (f_['dq'].data |
                    self.cleaned.crmask.astype(int) * 4096).astype('int16')
        f_.writeto(outfile, overwrite=overwrite)
