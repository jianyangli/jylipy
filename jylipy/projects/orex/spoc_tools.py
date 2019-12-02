import urllib.request, urllib.error, urllib.parse, http.cookiejar, sys, json, os, threading, queue
from datetime import datetime

user_jyli = 'jyli'
pwd_jyli = 'VUy-cw6-mDA-VLz'


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
        return response.read()
    except urllib.error.HTTPError as e:
        print('Failed to download. \nReason: %s\nQuery: %s' % (e.read(),query))


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
    instruments = ['ovirs','navcam']
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
        query_str = '{ "table_name": '+self.table_name+', "columnsaa": ["*"] }'
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
