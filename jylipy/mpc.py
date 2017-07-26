# Package to support Minor Planet Center interface and data processing
#
# 1/31/2017, JYL @PSI

from astropy.io import ascii
from . import Table, Column

def parse_mpcdata(infile, objtype='comet', decimal=False):
    from astropy.io import ascii
    from jylipy import Table, Column
    if objtype == 'comet':
        col = 'number', 'orbtype', 'desig', 'unused1'
        start = 0, 4, 5, 12
        end = 3, 4, 11, 12
        flag = 'flag',
    elif objtype == 'asteroid':
        col = 'number', 'desig', 'discovery'
        start = 0, 5, 12
        end = 4, 11, 12
        flag = 'band',
    elif objtype == 'satellite':
        col = 'pid', 'number', 's', 'desig', 'unused1'
        start = 0, 1, 4, 5, 12
        end = 0, 3, 4, 11, 12
        flag = 'band',
    else:
        ValueError('unsupported object type: ', objtype)
    col = col + ('note1', 'note2', 'date', 'ra', 'dec', 'unused2', 'mag') + flag + ('unused3', 'obscode',)
    start = start + (13, 14, 15, 32, 44, 56, 65, 70, 71, 77)
    end = end +     (13, 14, 31, 43, 55, 64, 69, 70, 76, 79)
    data = ascii.read(infile, format='fixed_width_no_header', names=col, col_starts=start, col_ends=end)
    data = Table(data)

    dstr = []
    ra1 = []
    dec1 = []
    for d in data:
        date = d['date'].split()
        fday = float(date[2])
        day = int(fday)
        hh = (fday-day)*24.
        mm = (hh-int(hh))*60.
        ss = (mm-int(mm))*60.
        day = '%02i' % day
        hh = '%02i' % int(hh)
        mm = '%02i' % int(mm)
        ss = '%06.3f' % ss
        dstr.append('T'.join(['-'.join(date[0:2]+[day]), ':'.join([hh,mm,ss])]))
        if decimal:
            ra = [float(x) for x in d['ra'].split()]
            ra1.append((ra[0]+ra[1]/double(60.)+ra[2]/double(3600.))*15.)
            dec = [float(x) for x in d['dec'].split()]
            dec1.append(dec[0]+dec[1]/double(60.)+dec[2]/double(3600.))

    k = list(data.keys())
    index = k.index('date')
    data.remove_column('date')
    data.add_column(Column(dstr, name='date'), index=index)
    #data['date'] = dstr
    #return ra1, dec1
    if decimal:

        index = k.index('ra'), k.index('dec')-1
        print(index)
        data.remove_columns(['ra','dec'])
        data.add_columns([Column(ra1, name='ra'),Column(dec1, name='dec')],indexes=index)

    return data
