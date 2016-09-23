#!/usr/bin/python

import datetime
import numpy as np
import matplotlib.finance as finance
import matplotlib.mlab as mlab
import pdb

startdate=datetime.date(2016,1,1)
today=enddate=datetime.date.today()
ticker='ngvc'

fh=finance.fetch_historical_yahoo(ticker,startdate,enddate)
r=mlab.csv2rec(fh)
r.sort()



pdb.set_trace()
