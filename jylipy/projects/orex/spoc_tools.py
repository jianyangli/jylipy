import urllib.request, urllib.error, urllib.parse, http.cookiejar, sys, json, os, threading, queue, re
from datetime import datetime
from ...apext import Time
from ...core import time_stamp


user_jyli = 'jyli'
pwd_jyli = 'giUjMTSJf4g8DcE'


def session_login(host, username=None, password=None):
    if username is None:
        username = input('Username: ')
    if password is None:
        from getpass import getpass
        password = getpass('Password: ')
    url = 'https://%s.lpl.arizona.edu/session-login' % host
    #Create the data to be sent and set the request header
    params = ('{"username": "%s","password":"%s"}' % (username,password)).encode()
    header = {'Content-Type':'application/json'}
    #Attempt to login:
    request = urllib.request.Request(url,params,header)
    try:
        response = urllib.request.urlopen(request)
        #Save the cookie for later
        cookiejar = http.cookiejar.LWPCookieJar('%s_cookie' % host)
        cookiejar.extract_cookies(response,request)
        cookiejar.save('%s_cookie'%(host))
        print('Successfully logged in')
    except urllib.error.HTTPError as e:
        print('Failed to login.\nReason: %s' % e)
        raise urllib.error.HTTPError


def webapi_post(host, query, endpoint, username=None, password=None):
    url = 'https://%s.lpl.arizona.edu/%s' % (host,endpoint)
    header = {'Content-Type':'application/json'}
    request = urllib.request.Request(url,query.encode(),header)
    if host+'_cookie' not in os.listdir('.'):
        session_login(host, username=username, password=password)
    elif (datetime.now() - datetime.fromtimestamp(os.path.getmtime(host+'_cookie'))).days > 2:
        session_login(host, username=username, password=password)
    cookiejar = http.cookiejar.LWPCookieJar('%s_cookie' % host)
    cookiejar.load('%s_cookie' % host)
    cookiejar.add_cookie_header(request)
    try:
        response = urllib.request.urlopen(request)
        return response.info(), response.read()
    except urllib.error.HTTPError as e:
        print('Failed to download. \nReason: %s\nQuery: %s' % (e.read(),query))
        return None, None


CONTENT_DISPOSITION_RE = re.compile(r'attachment; filename="(.*)"')
def parse_content_disposition(val):
    if not val:
        return None

    m = re.match(CONTENT_DISPOSITION_RE, val)
    if not m:
        return None

    return m.group(1)


class SPOC():
    def __init__(self, host='spocflight', username=user_jyli,
                password=pwd_jyli):
        self.host = host
        self.username = username
        self.password = password

    def login(self):
        session_login(self.host, self.username, self.password)

    def webapi_post(self, query, endpoint):
        return webapi_post(self.host, query, endpoint, username=self.username,
                password=self.password)


class DownloadHelper(threading.Thread):
    def __init__(self, download_queue, username=None, password=None):
        threading.Thread.__init__(self)
        self.download_queue = download_queue
        self.username=username
        self.password=password

    def run(self):
        while 1:
            (level,id) = self.download_queue.get()
            if level == 'l0':
                query = '{ "product": "ovirs_sci_frame.fits-l0-spot", "id": "ovirs_sci_frame.%s" }' % str(id)
                spot = webapi_post('spocflight',query,'data-get-product',username=self.username, password=self.password)
                with open('fits_l0_spot/ovirs_sci_frame_%s.fits' % str(id), 'wb') as f:
                    f.write(spot)
                    f.close()
            else:
                query = '{ "product": "ovirs_sci_level2_record.fits-l2-spot", "id": "ovirs_sci_level2_record.%s" }' % str(id)
                spot = webapi_post('spocflight',query,'data-get-product', username=self.username, password=self.password)
                with open('fits_l2_spot/ovirs_sci_level2_record_%s.fits' % str(id), 'wb') as f:
                    f.write(spot)
            self.download_queue.task_done()


def spoc_product_name(inst, level, group, cat='sci'):
    """Return the SPOC product name for the given instrument, level and group

    For OVIRS data:
        'l0_spot'      : 'ovirs_sci_frame'
        'l0_multipart' : 'ovirs_sci_level0'
        'l2_spot'      : 'ovirs_sci_level2_record'
        'l2_multipart' : 'ovirs_sci_level2'
        'l3a_spot'     : 'ovirs_sci_level3a_record'
        'l3a_multipart': 'ovirs_sci_level3a'
        'l3b_spot'     : 'ovirs_sci_level3b_record'
        'l3b_multipart': 'ovirs_sci_level3b'
        'l3e1'         : 'ovirs_sci_level3e_record'
        'l3e2'         : 'ovirs_sci_level3e'
    """
    instruments = ['ovirs']
    categories = ['sci']
    levels = ['l0', 'l2', 'l3a', 'l3b', 'l3c', 'l3e', 'l3f', 'l3g']
    groups = ['spot', 'multipart']

    if inst not in instruments:
        raise ValueError('instrument not recognized in {0}: {1}'.format(instruments, inst))
    if level not in levels:
        raise ValueError('level not recognized in {0}: {1}'.format(levels, level))
    if group not in groups:
        raise ValueError('group not recognized in {0}: {1}'.format(groups, group))
    if cat not in categories:
        raise ValueError('category not recognized in {0}: {1}'.format(categories, cat))

    if group.strip() == 'spot':
        grp_str = 'record'
    else:
        grp_str = ''

    lvl_str = level[0]+'evel'+level[1:]
    table_name = '_'.join([inst, cat, lvl_str, grp_str])

    if level == 'l0' and group == 'spot':
        table_name = '_'.join([inst, cat, 'frame'])
    table_name = '"'+table_name.strip('_')+'"'

    return table_name


class SPOC_Downloader():
    def __init__(self, inst, level, group, cat='sci', host='spocflight', user=user_jyli, passwd=pwd_jyli):
        self.table_name = spoc_product_name(inst, level, group, cat=cat)
        self.host = host
        self.instrument = inst
        self.level = level
        self.group = group
        self.category = cat
        self.username = user
        self.password = passwd

    def query(self):
        query_str = '{ "table_name": '+self.table_name+', "columns": ["*"] }'
        result = webapi_post(self.host, query_str, 'data-get-values', username=self.username, password=self.password)
        if result is None:
            return False
        self.data = json.loads(result)['result']
        return True

    def download(self, block=False, nworker=8):
        download_queue = query.Queue()
        for i in range(nworker):
            helper = DownloadHelper(download_queue, username=self.username, password=self.password)
            helper.daemon = True
            helper.start()
        for d in self.data:
            download_queue.put((self.level, d['id']))
        if block:
            download_queue.join()


class NAVCAM_Downloader(SPOC):
    """NAVCAM image downloader
    """
    def __init__(self, utc_start, utc_end, level, datadir=None):
        """
        utc_start, utc_end : str
            Start and end UTC
        level : in ['l0', 'l3a', 'l3b', 'l3ab']
            The level of data to be downloaded
        datadir : str, optional
            The directory that the downloaded data will be saved to.  Default
            is 'navcam_`date_string`/' in the the current working directory.
        user : str
            Login username
        passwd : str
            Login password
        """
        super().__init__()
        self.utc_start = utc_start
        self.utc_end = utc_end
        self.level = level
        if datadir is None:
            datadir = 'navcam_'+time_stamp(format=1).replace(':',
                    '').replace('-','')
        self.datadir = datadir
        if self.level not in ['l0', 'l3a', 'l3b', 'l3ab']:
            raise ValueError('Invalid data level specified.')

    def query(self):
        self._query_results = None
        if self.level == 'l0':
            query = '{"table_name": "navcam_image_header", "columns": ["id"], "time_utc": {"start": "%s", "end": "%s"} }' % (self.utc_start, self.utc_end)
            _, result = self.webapi_post(query, 'data-get-values')
        else:
            query = '{"columns": ["navcam_image_level3.id"], "time_utc": {"start": "%s", "end": "%s"} }' % ((self.utc_start, self.utc_end))
            _, result = self.webapi_post(query, 'data-get-values-related')

        result = json.loads(result)
        self._query_result = result
        return True

    def download(self, datadir=None):
        from os.path import isdir, join
        from os import mkdir
        if (not hasattr(self, '_query_result')) or (self._query_result is None):
            self.query()
        if datadir is None:
            datadir = self.datadir
        if self.level == 'l0':
            query = '{"product": "navcam_image_header.fits-opnav", "id": "navcam_image_header.%s"}'
            field = 'id'
        else:
            if self.level == 'l3a':
                query = '{"product": "navcam_image_level3.fits-l3a", "id": "navcam_image_level3.%s"}'
            elif self.level == 'l3b':
                query = '{"product": "navcam_image_level3.fits-l3a", "id": "navcam_image_level3.%s"}'
            elif self.level == 'l3ab':
                query = '{"product": "navcam_image_level3.fits-l3ab", "id": "navcam_image_level3.%s"}'
            field = 'navcam_image_level3.id'
        for id in self._query_result['result']:
            headers, data = self.webapi_post((query % str(id[field])),
                    'data-get-product')

            if type(data) != type(None):
                if not isdir(datadir):
                    mkdir(datadir)
                file_name = parse_content_disposition(headers.get(
                        'content-disposition'))
                if not file_name:
                    file_name = 'navcam_image_{}_{}.fits'.format(self.level,
                            str(id['field']))
                file_name = join(self.datadir, file_name)
                with open(file_name, 'wb') as output:
                    output.write(data)
                    output.close()
            else:
                print('Error downloading image %d' % id[field])

