#!/Users/jbungener/anaconda/bin/python
#!##/usr/bin/python


import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pdb
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
    
from get_google_data import get_google_data
import FinanceFunctions2 as ff2
import os


pwd=os.getcwd()

startDate = datetime.date(2015, 5, 1)# Date I started
today = endDate = datetime.date.today()
tickers = 	['NGVC']#,'CMG','BRK-B','ATRO'  ,'BIDU'  ,'ANIK' ,'RHT',]#'WETF','FRAN','LL','EGOV']
exch=		['NYSE']#,'NYSE','NYSE','NASDAQ','NASDAQ','NASDAQ','NYSE']# This much match the symbol above it. 

# these are the constants required by the various tools.
macdFast=8 #Rule#1:8   def:12
macdSlow=17 #Rule#1:17   def:26
macdSmoothing=9 #Rule#1:9   def:9
stoLength=14
#stoKPeriods=1#%K period
stoDPeriods=5 # %D periods Rule#1 recommends 5, default is 3
movAvePeriods=10 # number of periods for a moving average
fs=6
period=1*60*60# every 1 hours in seconds
window=1000# Number of days back to go. This forces Google to use the most possible (1 entire year back)


if macdFast>macdSlow:
	exit ('ERROR: MACDFast>MACDSlow')

for ticIdx in range(np.size(tickers)):

	finData=get_google_data(tickers[ticIdx],period,window,exch[ticIdx])

	timeDelt=finData.index[-1]-finData.index[1]
	if timeDelt.days<=(macdSlow+macdSmoothing+1): # not enough data to calculate the MACD
		exit (sprintf('ERROR: not enough days to calculate the MACD timeDelta=:%i days',int(timeDelt.days)))
		
		
 #MACD, EMAfast, EMAslow, MACDsign, MACDdiff=ff2.MACD(finData, n_slow, n_fast,macdSmoothing):  
	rule1Orders,goldOrders,silverOrders,macdDiff,fastStok,stoD,ma,ma20,ma50,ma200=ff2.buyOrSell(finData,macdFast,macdSlow,macdSmoothing,stoLength,stoDPeriods,movAvePeriods)
	
	latestMacdDiff=np.array(macdDiff)[-1]
	latestStok=fastStok[-1]
	latestSto=stoD[-1]
	latestMa=ma[-1]
	
	#fig , ax1 = plt.figure(facecolor='white')
	fig, axarr = plt.subplots(4, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].set_title(tickers[ticIdx]+' Last Date:'+finData.index[-1].strftime('%Y-%m-%d'))
	axarr[0].plot(finData.index,finData.close,'b',label='Close')
	axarr[0].plot(finData.index,ma,'g',label='%i day MA' % (movAvePeriods))
	axarr[0].plot(finData.index,ma20,'r',label='20 day MA')
	axarr[0].plot(finData.index,ma50,'k',label='50 day MA')
	axarr[0].plot(finData.index,ma200,'y',label='200 day MA')
	axarr[0].legend(loc=3,fontsize=fs)
	# Make the y-axis label and tick labels match the line color.
	axarr[0].set_ylabel('closing price & MAs',color='b')
	for tl in axarr[0].get_yticklabels(): tl.set_color('b')

	axarr[1].bar(finData.index,macdDiff,color='k')
	# Make the y-axis label and tick labels match the line color.
	axarr[1].set_ylabel('MACDdiff',color='k')
	for tl in axarr[1].get_yticklabels(): tl.set_color('k')
	
	axarr[2].plot(finData.index,fastStok,color='k',label='fastSTO %K')
	axarr[2].plot(finData.index,stoD,color='g',label='STO %D pers:'+str(stoDPeriods))
	# Make the y-axis label and tick labels match the line color.
	axarr[2].set_ylabel('STO%K and STO %d',color='k')
	axarr[2].legend(loc=2,fontsize=fs)
	
    
	    
	axarr[3].plot(finData.index,rule1Orders,'r')
	# Make the y-axis label and tick labels match the line color.
	axarr[3].set_ylabel('buy or sell',color='r')
	for tl in axarr[3].get_yticklabels(): tl.set_color('r')
	
	
	# Write a box with today's order
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	if rule1Orders[-1] == 1: textStr= 'Buy'
	elif rule1Orders[-1] == -1: textStr='Sell'
	else: textStr='Do Nothing'
	latestStr='\n latest MacdDiff is %.2f \n latest stoK is %.2f \n latest stoD is %.2f \n latest Moving Ave is %.2f \n latest price is %.2f' % (latestMacdDiff,latestStok, latestSto, latestMa, finData.close[-1])
	textString="""Current order is to """+textStr+ latestStr

	axarr[0].grid('on')
	axarr[1].grid('on')
	axarr[2].grid('on')
	axarr[3].grid('on')
	axarr[0].text(0.05, 0.95, textString, transform=axarr[0].transAxes, fontsize=8,
        verticalalignment='top', bbox=props)


	plt.savefig(pwd+'/'+tickers[ticIdx]+'_computerOrders.pdf')
	plt.close()
	
	#######################################################################################
	# Now we plot only the X last days
	daysBack=20
	
	mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
	alldays = DayLocator()              # minor ticks on the days
	weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
	dayFormatter = DateFormatter('%d')      # e.g., 12
	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.2)
	ax.xaxis.set_major_locator(mondays)
	#ax.xaxis.set_minor_locator(alldays)
	ax.xaxis.set_major_formatter(weekFormatter)
	
	#Another way of getting stock data.
	quotes = quotes_historical_yahoo_ohlc(tickers[ticIdx], startDate, today) # output format is: [datenum, open,high,low,close,volume] one date per row. 
	
	candlestick_ohlc(ax, quotes[-daysBack:-1],width=0.5)
	ax.plot(finData.index[-daysBack:-1],finData.close[-daysBack:-1],'b',label='Close')
	ax.plot(finData.index[-daysBack:-1],ma50[-daysBack:-1],'k',label='50 day MA')
	ax.set_title(tickers[ticIdx]+' Last Date:'+finData.index[-1].strftime('%Y-%m-%d'))
	ax.legend(loc='best',fontsize=fs) 
	ax.grid('on')
	ax.tick_params(labelsize=fs)
	ax.xaxis.label.set_fontsize(fs)
	plt.savefig(pwd+'/'+tickers[ticIdx]+'_CandlestickPast%iDays.pdf' %daysBack)
	plt.close()
	#dateOordinal=[0 for i in range(np.size(finData.index[-daysBack:-1]))] # I need dates as a number
	#quotes2=[[] for i in range(np.size(finData.index[-daysBack:-1]))] # I need dates as a number
	#for dateIdx in range (np.size(finData.index[-daysBack:-1])): 
		#dateOordinal[dateIdx]=finData.index[dateIdx].toordinal()
		#quotes2[dateIdx]=[finData.index[dateIdx].toordinal(), finData.open[dateIdx],finData.high[dateIdx],finData.low[dateIdx],finData.close[dateIdx],finData.volume[dateIdx]]
	

	#ax.xaxis_date()
	#ax.autoscale_view()
	#plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
	#plt.show()
	
	#pdb.set_trace()
	fig, axarr = plt.subplots(4, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].set_title(tickers[ticIdx]+' Last Date:'+finData.index[-1].strftime('%Y-%m-%d'))
	axarr[0].plot(finData.index[-daysBack:-1],finData.close[-daysBack:-1],'b',label='Close')
	axarr[0].plot(finData.index[-daysBack:-1],ma[-daysBack:-1],'g',label='%i day MA' % (movAvePeriods))
	#axarr[0].plot(finData.index[-daysBack:-1],ma20[-daysBack:-1],'r',label='20 day MA')
	axarr[0].plot(finData.index[-daysBack:-1],ma50[-daysBack:-1],'k',label='50 day MA')
	#axarr[0].plot(finData.index[-daysBack:-1],ma200[-daysBack:-1],'y',label='200 day MA')
	axarr[0].legend(loc=3,fontsize=fs)
	axarr[0].set_ylabel('closing price & MAs',color='b')
	for tl in axarr[0].get_yticklabels(): tl.set_color('b')

	axarr[1].bar(finData.index[-daysBack:-1],macdDiff[-daysBack:-1],color='k')
	# Make the y-axis label and tick labels match the line color.
	axarr[1].set_ylabel('MACDdiff',color='k')
	for tl in axarr[1].get_yticklabels(): tl.set_color('k')
	
	axarr[2].plot(finData.index[-daysBack:-1],fastStok[-daysBack:-1],color='k',label='fastSTO %K')
	axarr[2].plot(finData.index[-daysBack:-1],stoD[-daysBack:-1],color='g',label='STO %D pers:'+str(stoDPeriods))
	# Make the y-axis label and tick labels match the line color.
	axarr[2].set_ylabel('STO%K and STO %d',color='k')
	axarr[2].legend(loc=2,fontsize=fs)   
	    
	axarr[3].plot(finData.index[-daysBack:-1],rule1Orders[-daysBack:-1],'r')
	# Make the y-axis label and tick labels match the line color.
	axarr[3].set_ylabel('Rule #1 buy or sell',color='r')
	axarr[3].set_ylim([-1.1,1.1])
	for tl in axarr[3].get_yticklabels(): tl.set_color('r')
	
	axarr[0].grid('on')
	axarr[1].grid('on')
	axarr[2].grid('on')
	axarr[3].grid('on')
	plt.savefig(pwd+'/'+tickers[ticIdx]+'_past%iDays.pdf' %daysBack)
	plt.close()
	
	#########################################################################################
	
	#Now we look at the performance in the past, since the dates entered above. 
	moneyStart=10000# Dollars to start with
	commission=8.99 # transaction dollars	
	
	netWorthRule1,actionsRule1 = ff2.backTest(rule1Orders,finData, moneyStart,commission)	# Rule 1
	netWorthGC,actionsGC = ff2.backTest(goldOrders,finData, moneyStart,commission)	 # Gold Cross
	netWorthSC,actionsSC = ff2.backTest(silverOrders,finData, moneyStart,commission)	 # Silver Cross
			
	
	fig, axarr = plt.subplots(3, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].plot(finData.index,finData.close,'b',label='Close')
	axarr[0].plot(finData.index,ma,'g',label='%i day MA' % (movAvePeriods))
	axarr[0].plot(finData.index,ma20,'r',label='20 day MA')
	axarr[0].plot(finData.index,ma50,'k',label='50 day MA')
	axarr[0].plot(finData.index,ma200,'y',label='200 day MA')
	axarr[1].plot(finData.index,netWorthRule1,'k',label='netWorth Rule1')
	axarr[1].plot(finData.index,netWorthGC,'y',label='netWorth Gold Cross')
	axarr[1].plot(finData.index,netWorthSC,'g',label='netWorth Silver Cross')
	axarr[2].plot(finData.index,actionsRule1,'ok',label='Rule1 Action taken')
	axarr[2].plot(finData.index,actionsGC,'*y',label='Gold Cross Action taken')
	axarr[2].plot(finData.index,actionsSC,'sg',label='Silver Cross Action taken')
	axarr[2].plot(finData.index,rule1Orders,'k',label='Rule 1 Order')
	axarr[2].plot(finData.index,goldOrders,'y',label='Gold Cross Order')
	axarr[2].plot(finData.index,silverOrders,'g',label='Silver Cross Order')
	axarr[2].set_ylim([-1.1,1.1])
	strTitle=tickers[ticIdx]+' with result Rule1=%i, GoldCross=%i, SilverCross=%i, started with %i'% (netWorthRule1[-1],netWorthGC[-1],netWorthSC[-1],moneyStart)
	axarr[0].set_title(strTitle)
	axarr[0].grid('on')
	axarr[1].grid('on')
	axarr[2].grid('on')
	
	axarr[0].set_ylabel('Closing Price')
	axarr[1].set_ylabel('NetWorth')
	axarr[2].set_ylabel('Action taken, Orders, sell==-1, Do nothing==0, Buy ==1')
	axarr[0].legend(loc=3,fontsize=fs)
	axarr[1].legend(loc=6,fontsize=fs)
	axarr[2].legend(loc=6,fontsize=fs)
	plt.savefig(pwd+'/'+tickers[ticIdx]+'_MoneyMade.pdf')
	plt.close()

	