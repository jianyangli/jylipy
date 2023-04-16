import numpy as np, telnetlib, requests, astropy.units as u
from astropy.table import Table, QTable
from astropy.time import Time
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from astroquery.jplhorizons import Horizons, conf
from .core import is_iterable


class JPLHorizons(telnetlib.Telnet):
    '''JPL Horizons telnet class'''

    host = 'horizons.jpl.nasa.gov'
    port = 6775

    def __init__(self):
        telnetlib.Telnet.__init__(self, self.host, port=self.port)


def getsboscelt(name, recno=None, spice=False, disp=True):
    '''Retrieve osculation elements of small bodies from JPL Horizons

 Parameters
 ----------
 name : str
   Name of the small body
 recno : str, int, optional
   When orbital elements in multiple epochs are found (e.g., comets),
   `recno` is used to select the epoch.
 spice : bool, optional
   If `True`, then program returns parameters in the format that NAIF
   SPICE toolkit would like to take.  See, e.g., spice.conics,
   spice.oscelt.  Otherwise program returns a dictionary.
 disp : bool, optional
   If `True`, the information returned from JPL Horizons will be
   printed out on screen.  `False` will suppress the output.

 Returns
 -------
 A dictionary or a tuple containing the osculation elements of `name`
 The parameters are:

 If `spice` = `True`:
   Perihelion (km)
   Eccentricity (rad)
   Inclination (rad)
   Longitude of ascending node (rad)
   Argument of perihelion (rad)
   Mean anomaly at epoch (rad)
   Epoch (ephemeris seconds past J2000)

 If `spice` = `False:
   'A' : Semimajor axis (AU)
   'ADIST' : Aphelion distance (AU)
   'EC' : Eccentricity
   'EPOCH' : Epoch at which the elements are defined (JD)
   'IN' : Inclination (deg)
   'MA' : Mean anomaly (deg)
   'N' : Mean motion (deg/d)
   'OM' : Longitude of ascending node (deg)
   'PER' : Period (years)
   'QR' : Perihelion (AU)
   'TP' : Time of perihelion (JD)
   'W' :Argument of perihelion (deg)

 Notes
 -----
 Program uses telnet service telnet: ssd.jpl.nasa.gov:6775 to access
 the orbital elements.

 v1.0.0 : JYL @PSI, 6/6/2014
    '''

    import astropy.units as u

    if disp:
        print('Initializing telnet ...')
        print()
    import telnetlib
    tn = JPLHorizons()
    tn.read_until(b'Horizons>')
    tn.write((name+'\n\n').encode('ascii'))
    ret = tn.read_until(b'<cr>:')
    ret = ret[ret.find(b'**'):ret.rfind(b'**')]
    if ret[:160].lower().find(name.lower().encode('ascii')) < 0:  # not find
        if ret.find(b'No matches found') > 0:  # not found
            print((name+': Object not found.'))
            tn.write(b'exit\n')
            return
        else:  # multiple matches found
            if recno is None:   # record number not specified
                print('Multiple records found.  Please specify record number.')
                print('  get_oscelt(name, recno=#)')
                print()
                for s in ret.split(b'\r\n'):
                    print(s)
                tn.write(b'exit\n')
                return
            else:  # use record number
                tn.write((str(recno)+'\n\n').encode('ascii'))
                ret = tn.read_until(b'<cr>:')
                ret = ret[ret.find(b'**'):ret.rfind(b'**')]
                if ret[:160].lower().find(name.lower()) < 0:
                    print((name+': Object not found.'))
                    tn.write(b'exit\n')
                    return
    tn.write(b'exit\n')
    if disp:
        for s in ret.split(b'\r\n'):
            print(s.decode('utf-8'))

    # extract parameters
    if disp:
        print()
        print('Extracting osculation elements ...')
    keys = [b' '+s for s in b'EPOCH= ! EC= QR= TP= OM= W= IN= A= MA= ADIST= PER= N= ANGMOM= DAN= DDN= L= B= MOID= TP='.split()]
    offset = [len(k)-1 for k in keys]
    ind = [ret.find(s)+1 for s in keys]
    ind[-1] = ret.rfind(keys[-1])
    par = [ret[(ind[i]+offset[i]):ind[i+1]] for i in range(len(ind)-1)]
    par.pop(1)
    elm = [float(p.split()[0]) for p in par]

    # return results
    if disp:
        print()
        print('Done.')
    if spice:
        import spiceypy as spice
        ep = spice.str2et('JD '+str(elm[0])) - spice.str2et('2000-01-01.0')
        return u.au.to(u.km, elm[2]),elm[1],np.deg2rad(elm[6]),np.deg2rad(elm[4]),np.deg2rad(elm[5]),np.deg2rad(elm[8]),ep
    else:
        return {'QR':elm[2],'EC':elm[1],'IN':elm[6],'OM':elm[4],'W':elm[5],'MA':elm[8],'EPOCH':elm[0],'TP':elm[3],'A':elm[7],'ADIST':elm[9],'PER':elm[10],'N':elm[11]}


def getsbspk(name, start, stop, outfile, recno=None, overwrite=False, verbose=True):
    '''Retrieve SPK of small bodies from JPL Horizons

 Parameters
 ----------
 name : str, or array-like str
   Name(s) of small bodies to be retrieved
 start, stop : str
   Start and stop date in JPL Horizons format (yyyy-mm-dd, yyy-mmm-dd,
   etc)
 outfile : str
   Output file.  All SPK will be merged to a single output file
 recno : int, str, or array-like, optional
   In case the targets have multiple entries (e.g., comets), this
   keyword contains the record number in JPL Horizons.  It has to have
   the same length as `name`.  The elements that are not needed are
   simply ignored.
 overwrite : bool, optional
   If `True`, then if output file exists, it will be overwritten.  If
   `False`, then if output file exists, an error message will be
   printed out.
 verbose : bool, optional
   Enable screen printout

 Returns
 -------
 None

 Notes
 -----
 Program uses telnet service telnet://ssd.jpl.nasa.gov:6775/ to
 generate SPK kernel file, then use ftp service to download it,
 and finally call SPICE tool `tobin` to convert to binary SPK kernel.

 v1.0.0, JYL @PSI, 6/6/2014.
    '''

    import os
    if os.path.isfile(outfile):
        if overwrite:
            os.remove(outfile)
        else:
            print('Output file exist.')
            return

    if isinstance(name, (str,bytes)):
        nms = [name]
        if recno is not None:
            recs = [recno]
    else:
        nms = name
        if recno is not None:
            recs = recno

    # init telnet
    if verbose:
        print('Initializing telnet://horizons.jpl.nasa.gov:6775/ ...')
    import telnetlib
    tn = JPLHorizons()
    tn.read_until(b'Horizons>')

    # adding objects into spk
    n = 0
    while True:
        if verbose:
            print(('Adding '+nms[n]+' ...'))
        tn.write((nms[n]+'\n\n').encode('ascii'))
        ret = tn.read_until(b'<cr>:')
        ret = ret[ret.find(b'**'):ret.rfind(b'**')]
        if ret[:160].lower().find(nms[n].lower().encode('ascii')) < 0:
            if ret.find(b'No matches found') > 0:
                print((nms[n]+': Object not found.  Skipped.'))
                n += 1
                if n == len(nms):
                    tn.write('f\n')
                    break
                continue
            else:  # multiple record found
                if recno is None:
                    print((nms[n]+': Multiple records found, no record number specified.  Skipped.'))
                    n += 1
                    if n == len(nms):
                        tn.write(b'f\n')
                        break
                    continue
                else:  # use record number
                    tn.write((str(recs[n])+'\n').encode('ascii'))
                    ret = tn.read_until(b'<cr>:')
                    ret = ret[ret.find(b'**'):ret.rfind(b'**')]
                    if ret[:160].lower().find(nms[n].lower().encode()) < 0:
                        print((nms[n]+': Error: Object not found.  Skipped.'))
                        n += 1
                        if n == len(nms):
                            tn.write(b'f\n')
                            break
                        continue

        tn.write(b's\n')
        if n == 0:
            tn.write(b'jyli@psi.edu\nyes\n\n')
        tn.write(b'B\n')
        tn.write((start+'\n'+stop+'\n').encode('ascii'))
        n += 1
        if n == len(nms):
            tn.write(b'no\n')
            break
        tn.write(b'yes\n')
        dmp = tn.read_until(b'Select ... [E]phemeris, [M]ail, [R]edisplay, ?, <cr>')  # dump output

    # extract output file name
    tn.write(b'exit\n')
    ret = tn.read_all()
    ind = ret.rfind(b'Full path')
    outstr = ret[ind:ind+80].split(b'\r\r\n')[0]
    out = outstr[outstr.find(b'ftp'):]

    # download output file
    if verbose:
        print(('Downloading SPK file from '+outstr.split()[3].decode('utf-8')+' ...'))
    import os
    spkfile = open(outfile, 'wb')
    import ftplib
    outstr = out.split(b'/')
    ftp = ftplib.FTP(outstr[2])
    ftp.login()
    ftp.cwd(b'/'.join(outstr[3:-1]).decode('utf-8'))
    ftp.set_pasv(True)
    ftp.sendcmd('OPTS UTF8 ON')
    ftp.sendcmd('TYPE I')
    ftp.retrbinary('RETR '+outstr[-1].decode('utf-8'), spkfile.write)
    ftp.quit()
    spkfile.close()

    if verbose:
        print('Done.')


class HorizonsError(Exception):
    def __init__(self, message='Horizons error'):
        self.message = message


def geteph(*args, **kwargs):
    '''Calculate ephemerides of target

    eph = geteph('ceres', '2015-01-01', '2015-12-31', '1d')
    eph = geteph('ceres', ['2015-01-01', '2015-01-02', ...])

    keywords:

    observatory_code : str or int, optional
        Observer's location code, default is '@399'
    id_type : str, optional
        Identifier type, see astroquery.jplhorizons.Horizons.__init__
    outfile : string, optional
        File name to save the ephemerides

    Returns an astropy Table with all ephemerides fields
    '''

    observatory_code = kwargs.pop('observatory_code', '@399')
    id_type = kwargs.pop('id_type', 'smallbody')
    outfile = kwargs.pop('outfile', None)

    nargs = len(args)
    if nargs not in [2, 4]:
        raise TypeError('geteph() takes 2 or 4 arguments ({0} given)'.format(nargs))

    if nargs == 2:
        jds = Time(args[1]).jd
        if is_iterable(args[1]):
            jds = list(jds)
        else:
            jds = [jds]
    else:
        jds = {'start': args[1], 'stop': args[2], 'step': args[3]}
    obj = Horizons(id=args[0], location=observatory_code, epochs=jds, id_type=id_type)
    try:
        out = obj.ephemerides()
        out = Table(out)
        if outfile is not None:
            out.write(outfile)
        return Table(out)
    except:
        raise HorizonsError('Horizons Error')



class SBDBCloseApproach():
    """Implementation of JPL SBDB Close-Approach Data API
    https://ssd-api.jpl.nasa.gov/doc/cad.html
    """
    # close approach data api url
    _api_url = "https://ssd-api.jpl.nasa.gov/cad.api"
    # input parameter table
    _parfile = 'cad_par.csv'
    # input parameter type look up table
    _type_lut = {'string': str,
                 'number': float,
                 'boolean': bool,
                 'int': int
                 }
    # output data field formats and units
    _output_types = {'des': (str, ''),
                     'orbit_id': (int, ''),
                     'jd': (float, ''),
                     'cd': (str, ''),
                     'dist': (float, 'au'),
                     'dist_min': (float, 'au'),
                     'dist_max': (float, 'au'),
                     'v_rel': (float, 'km/s'),
                     'v_inf': (float, 'km/s'),
                     't_sigma_f': (str, ''),
                     'body': (str, ''),
                     'h': (float, 'mag'),
                     'diameter': (float, 'km'),
                     'diameter_sigma': (float, 'km'),
                     'fullname': (str, '')
                    }

    def __init__(self):
        # set up query parameters
        fn = get_pkg_data_filename(self._parfile)
        self.parinfo = ascii.read(fn)
        tp = [self._type_lut[k] for k in self.parinfo['Type']]
        self.parinfo.remove_column('Type')
        self.parinfo.add_column(Column(tp, name='Type'), index=1)

    def __call__(self, **kwargs):
        self.query(**kwargs)
        if self.status_code == 200:
            return self.to_table()
        else:
            return None

    def query(self, **kwargs):
        """Query the SBDB Close-Approach Data

        Attributes added
        ----------------
        .query_par : dict
            Query parameters
        .status_code : int
            Status code of query
        .response : dict
            The JSON deserialized response from query, typically the output
            from `.query`.
        .count : int
            Number of objects found
        .signature : str
            Signature of query
        """
        # adjust key names
        kwargs_ = {}
        for k, v in kwargs.items():
            kwargs_[k if '_' not in k else k.replace('_', '-')] = v
        self.query_par = kwargs_
        # parse parameters
        pars = []
        for k, v in kwargs_.items():
            # check parameter name
            if k not in self.parinfo['Parameter']:
                raise ValueError('parameter {} undefined.'.format(k))
            # check parameter value
            row_index = np.where(self.parinfo['Parameter'] == k.lower())[0][0]
            if not isinstance(v, self.parinfo[row_index]['Type']):
                raise ValueError('parameter {} is of type {}, {} is required.'.
                    format(k, type(v), self.parinfo[row_index]['Type']))
            pars.append('{}={}'.format(k, v))
        url = '?'.join([self._api_url, '&'.join(pars)])
        resp = requests.get(url)
        out = resp.json()
        self.status_code = resp.status_code
        if resp.status_code == 200:
            # success
            self.response = out
            self.count = self.response['count']
            self.signature = self.response['signature']
        else:
            # bad request
            print('Query failed with status code {}\n{}\nSee {} for more'
                  ' information.'.format(out['code'], out['message'],
                                         out['moreInfo']))

    def to_table(self):
        """Parse the query results into a table

        Returns
        -------
        `astropy.table.QTable`, the query results
        """
        if self.count > 0:
            dtypes = [self._output_types[k][0] for k in
                            self.response['fields']]
            meta = self.signature.copy()
            meta.update(self.query_par)
            out = QTable(names=self.response['fields'], dtype=dtypes,
                    meta=meta)
            for row in self.response['data']:
                out.add_row(row)
            for k in self.response['fields']:
                if (self._output_types[k][0] != str and
                    self._output_types[k][1] != ''):
                    out[k].unit = u.Unit(self._output_types[k][1])
            return out
        else:
            return None

    def write(self, outfile, **kwargs):
        """Write query results to file.

        The data can be saved in either ascii format or fits binary table.
        The output format is either specified by the file name extension,
        or by keywords.

        Parameters
        ----------
        outfile : str
            Name of output file
        kwargs : dict
            Keyword parameters accepted by `astropy.table.Table.write()`
        """
        if getattr(self, 'status_code', None) is None:
            print('No query has been made.  No file is written.')
            return
        if getattr(self, 'count', 0) == 0:
            print('No entry is returned from query.  No file is written.')
            return
        tbl = self.to_table()
        tbl.write(outfile, **kwargs)
