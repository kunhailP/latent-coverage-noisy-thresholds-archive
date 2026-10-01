"""Download apipop from the CRAN `survey` package and save data/apipop.pkl.

The data are not redistributed in this repository.
"""
import io
import re
import tarfile
import urllib.request

import rdata
from _common import DATA

DATA.mkdir(exist_ok=True)
index = urllib.request.urlopen('https://cran.r-project.org/web/packages/survey/index.html').read().decode()
ver = re.search(r'survey_[0-9.\-]+\.tar\.gz', index).group(0)
raw = urllib.request.urlopen(f'https://cran.r-project.org/src/contrib/{ver}').read()
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    blob = tf.extractfile('survey/data/api.rda').read()
(DATA / 'api.rda').write_bytes(blob)
api = rdata.conversion.convert(rdata.parser.parse_file(DATA / 'api.rda'))
api['apipop'].to_pickle(DATA / 'apipop.pkl')
print(f'{ver}: apipop {api["apipop"].shape} -> data/apipop.pkl')
