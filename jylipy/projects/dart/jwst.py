# JWST image analysis

import numpy as np
from ...saoimage import getds9
from ...core import shift


class PSFRemoval():
    """PSF removal for JWST images"""

    def __init__(self, psf, center=None):
        """
        psf : 2d array
            The input PSF.  No need to be normalized or scaled.
        center : two-element array-like
            (y, x) center of the PSF array.  Default is at image center.
                center = (np.array(psf.shape) - 1) / 2
            The center of the bottom-left pixel is (0, 0)
        """
        self.PSF = psf
        self.psf_center = center if center is not None \
                            else (np.array(psf.shape) - 1) / 2

    def trial(self, image, center=None, scale=1., ds9=None):
        """Interactive facility to find the best parameter for PSF removal.

        This routine will take the input from the user to experiment with
        PSF removal until the best image is determined by the user.  Trial
        PSF subtracted image will be displayed in DS9 for visual inspection.
        The best parameters will be returned, including the center of the
        image and the scaling factor to be applied to the PSF before
        subtraction.

        """
        self.image = image
        center = (np.array(image.shape) - 1) / 2 if center is None \
                                                else np.array(center)

        # trim or pad PSF to be the same size and input image
        psf, psfct = self._trim_pad_psf(image)

        # prepare display
        d = getds9('psf_removal') if ds9 is None else ds9

        # trial loop
        key = ''
        step = 1
        dscl = 0.1
        while key != 'q':
            psf_s = shift(psf, center - psfct)
            cln_img = image - psf_s * scale
            if d.get('frame') == '':
                d.imdisp(cln_img)
            else:
                pan = d.get('pan')
                d.imdisp(cln_img, newframe=False)
                d.set('pan to {}'.format(pan))
            key = input()
            if key == 'a':
                # shift left
                center[1] -= step
            elif key == 's':
                # shift down
                center[0] -= step
            elif key == 'd':
                # shift right
                center[1] += step
            elif key == 'w':
                # shift up
                center[0] += step
            elif key == 'c':
                # set shift step size
                while True:
                    key = input('new shift step size: ')
                    try:
                        step = float(key)
                        break
                    except:
                        print('wrong input, try again.')
            elif key == 'z':
                # twice shift step size
                step *= 2
                print('step size {:.2f}'.format(step))
            elif key == 'x':
                # half shift step size
                step /= 2
                print('step size {:.2f}'.format(step))
            elif key == 't':
                # set scaling factor
                while True:
                    key = input('new scaling factor: ')
                    try:
                        scale = float(key)
                        break
                    except:
                        print('wrong input, try again.')
                print('scaling factor is set to {:.2f}'.format(scale))
            elif key == 'r':
                # set scaling factor
                while True:
                    key = input('new scaling step: ')
                    try:
                        dscl = float(key)
                        break
                    except:
                        print('wrong input, try again.')
                print('scaling step {:.2f}'.format(dscl))
            elif key == 'i':
                # twice scaling factor
                dscl *= 2
                print('scaling step {:.2f}'.format(dscl))
            elif key == 'o':
                # half scaling factor
                dscl /= 2
                print('scaling step {:.2f}'.format(dscl))
            elif key == 'y':
                # increase scaling factor
                scale += dscl
                print('scaling factor {:.2f}'.format(scale))
            elif key == 'u':
                # decrease scaling fator
                scale -= dscl
                print('scaling factor {:.2f}'.format(scale))
            elif key == 'p':
                print('scaling factor = {:.2f}, delta_scaling = {:.2f}'.
                    format(scale, dscl))
                print('shift step = {:.2f}'.format(step))
            elif key == 'h':
                print('shift image: a, s, d, w')
                print('change shift steps:')
                print('    double: z')
                print('    half: x')
                print('    set value: c')
                print('change scaling factor:')
                print('    increase: y')
                print('    decrease: u')
                print('    set scaling factor: t')
                print('    double step: i')
                print('    half step: o')
                print('    set scaling change step: r')
                print('print parameters: p')
                print('quit: q')
                print('')

        print('scaling factor = {:.2f}, delta_scaling = {:.2f}'.
                format(scale, dscl))
        print('shift step = {:.2f}'.format(step))

        self.scale = scale
        self.image_center = center

    def clean(self, image=None, scale=None, center=None):
        """Remove PSF to produce clean image.

        image : 2d array
            Image to be cleaned with PSF removal.  Default is `self.image`
        scale : float
            Scaling factor.  Default is `self.scale`
        center : 2-element array
            Center of image (y, x).  Default is `self.image_center`

        Return
        ------
        Cleaned image in a 2d array
        """
        image = getattr(self, 'image', None) if image is None else image
        if image is None:
            raise ValueError('no input image found.')
        scale = getattr(self, 'scale', None) if scale is None else scale
        if scale is None:
            raise ValueError('no scaling factor provided.')
        center = getattr(self, 'image_center', None) if center is None \
                            else np.array(center)
        if center is None:
            raise ValueError('no image center specified.')
        psf, psfct = self._trim_pad_psf(image)
        psf_s = shift(psf, center - psfct)
        return image - psf_s * scale

    def _trim_pad_psf(self, image):
        """Trim or pad PSF to be the same size as input image

        Returns the new psf and the center in a tuple
        """
        psf = self.PSF.copy()
        if self.PSF.shape == image.shape:
            return psf, self.psf_center
        ct = []  # center of psf after trimming or padding
        for i in [0, 1]:
            psf = np.rollaxis(psf, i, 0)
            ds = self.PSF.shape[i] - image.shape[i]
            if ds > 0:
                # crop
                sz2 = image.shape[i] // 2
                x1 = int(self.psf_center[i]) - sz2
                x2 = int(self.psf_center[i]) + sz2
                if image.shape[i] %2 != 0:
                    x2 += 1
                psf = psf[x1:x2]
                ct.append(sz2 + self.psf_center[i] % 1)
            elif ds < 0:
                # pad
                pw = -ds // 2
                pw = [pw, pw]
                if -ds % 2 != 0:
                    pw[1] += 1
                psf = np.pad(psf, pad_width=(list(pw), (0, 0)),
                             constant_values=0)
                ct.append(self.psf_center[i] + pw[0])
            psf = np.rollaxis(psf, 0, i+1)

        return psf, ct
