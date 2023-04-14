from astropy.nddata import CCDData
import ccdproc as ccdp

class CRRej():
    """Cosmic ray rejection from HST images"""
    
    def __init__(self, sigclip=4, cleantype='medmask', niter=10,
                 masks=[4, 256, 512]):
        """
        sigclip, cleantype, niter
            Parameters for `ccdproc.cosmicray_lacosmic`
        mask : list of int, optional
            The UVIS DQ mask values to be used for the cosmic ray
            rejection
        """
        self.sigclip = sigclip
        self.cleantype = cleantype
        self.niter = niter
        self.masks = masks
        
    def clean(self, infile, verbose=True):
        # load image
        im = CCDData.read(infile)
        with fits.open(infile) as f_:
            dq = f_['dq'].data
            rn = f_[0].header['PCTERNOI']
        
        # generate mask from dq array
        mask = np.zeros_like(dq)
        for v in self.masks:
            mask |= dq & v
        im.mask = mask > 0

        # remove cosmic ray
        self.cleaned = ccdp.cosmicray_lacosmic(im, readnoise=rn,
            sigclip=sigclip, cleantype=cleantype, niter=niter,
            verbose=verbose)

    def write(self, outfile, infile=None, overwrite=False):
        """
        If input file is provided, then the 'sci' and 'dq' segments will
        be updated with the cleaned image and cosmic ray mask.  Otherwise
        a new HDU list will be generated that contains the 'sci' and 'dq'
        segments.
        
        The 'dq' mask marks the cosmic ray affected pixels by 4096.
        """
        if infile is not None:
            f_ = fits.open(file)
        else:
            f_ = fits.HDUList([fits.PrimaryHDU()])
            f_.append(fits.ImageHDU(np.zeros_like(self.cleaned.data,
                        dtype='float32'), name='sci'))
            f_.append(fits.ImageHDU(np.zeros_like(self.cleaned.data,
                        dtype='int16'), name='dq'))
        f_['sci'].data = self.cleaned.data.astype('float32')
        crmask = self.cleaned.mask
        crmask[mask > 0] = False
        f_['dq'].data = (f_['dq'].data | crmask.astype(int) * 4096).astype(
            'int16')
        f_.writeto(outfile, overwrite=overwrite)

