# JWST/NIRCam related classes

from astropy.utils.data import get_pkg_data_filename
from astropy.io import ascii

class DQFlag():
    """Data quality flags
    """

    _dqflags_def = 'nircam_dqflags.csv'

    def __init__(self, nbit=32):
        fn = get_pkg_data_filename(self._dqflags_def)
        self.dqflag_def = ascii.read(fn)
        self.nbit = nbit

    def decode(self, flag):
        """Decode a flag, return a list of flag positions.

        Parameters
        ----------
        flag : int
            The data quality flag to be decoded

        Returns
        -------
        list : the flag positions in the input `flag`
        """
        pos = []
        mask = 1
        for i in range(self.nbit):
            if flag & mask:
                pos.append(i)
            mask <<= 1
        return pos

    def set(self, pos):
        """Set flag positions and return a flag

        Parameters
        ----------
        pos : int or list of int
            The positions to be set

        Returns
        -------
        int : The flag with `pos` set
        """
        flag = 0
        if not hasattr(pos, '__iter__'):
            pos = [pos]
        for p in pos:
            flag |= 1 << p
        return flag

    def merge(self, flags):
        """Merge multiple flags.

        Parameters
        ----------
        flags : int or list of int
            Input flags to be merged

        Returns
        -------
        int : Merged flag
        """
        if not hasattr(flags, '__iter__'):
            flags = [flags]
        merged = flags[0]
        for f in flags[1:]:
            merged |= f
        return merged

    def info(self, flag_pos=None):
        """Return the information of a flag or a set of flags.

        Parameters
        ----------
        flag_pos : int or list of int
            Positions of flags.  Default is all flags.

        Returns
        -------
        astropy.table.Table : Information table of flags
        """
        if flag_pos is None:
            return self.dqflag_def
        else:
            if not hasattr(flag_pos, '__iter__'):
                flag_pos = [flag_pos]
            out = self.dqflag_def[[]]
            for p in flag_pos:
                out.add_row(self.dqflag_def[self.dqflag_def['Bit'] == p][0])
            return out

