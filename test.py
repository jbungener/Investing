#!/Users/jbungener/anaconda/bin/python

#from yahoo_finance import Currency
#from yahoo_finance import Share
import pdb
from datetime import date
import datetime
import numpy as np
from get_google_data import get_google_data

#print (python.version)

startDate = '2015-05-01'# Date I started
startD=datetime.date(2015, 5, 1)
today = datetime.datetime.today().strftime('%Y-%m-%d')
tDelta=date.today()-date(2015,5,1)
numDays=1000 # maximum from Google is 350 days back
ticker=['NGVC']
exch=['NYSE']
period=4*60*60# every 4 hours in seconds
window= numDays
gData = get_google_data(ticker[0],period,window,exch[0])
pdb.set_trace()


#tick= Share(ticker[0])
#data=np.array(tick.get_historical(startDate, today))
 


