#!/Users/jbungener/Python/anaconda3.2/bin/python
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

#import FinanceFunctions as ff
import FinanceFunctions2 as ff2
import os


pwd=os.getcwd()

startDate = datetime.date(2015, 5, 1)# Date I started
today = endDate = datetime.date.today()
tickers = ['NGVC','CMG','BRK-B','ATRO','BIDU','ANIK','RHT','WETF','FRAN','LL','EGOV']


# these are the constants required by the various tools.
macdFast=8 #Rule#1:8   def:12
macdSlow=17 #Rule#1:17   def:26
macdSmoothing=9 #Rule#1:9   def:9
stoLength=14
#stoKPeriods=1#%K period
stoDPeriods=5 # %D periods Rule#1 recommends 5, default is 3
movAvePeriods=10 # number of periods for a moving average
fs=6


if macdFast>macdSlow:
	exit ('ERROR: MACDFast>MACDSlow')

for ticIdx in range(np.size(tickers)):

	fh = finance.fetch_historical_yahoo(tickers[ticIdx], startDate, endDate)
# a numpy record array with fields: date, open, high, low, close, volume, adj_close)
	finData = mlab.csv2rec(fh)
	fh.close()
	finData.sort()

	timeDelt=finData.date[-1]-finData.date[1]
	if timeDelt.days<=(macdSlow+macdSmoothing+1): # not enough data to calculate the MACD
		exit (sprintf('ERROR: not enough days to calculate the MACD timeDelta=:%i days',int(timeDelt.days)))
		
		
 #MACD, EMAfast, EMAslow, MACDsign, MACDdiff=ff2.MACD(finData, n_slow, n_fast,macdSmoothing):  
	rule1Orders,ordersTechnik1,ordersTechnik2,priceTechnik2,macdDiff,fastStok,stoD,ma,ma20,ma50,ma200=ff2.buyOrSell(finData,macdFast,macdSlow,macdSmoothing,stoLength,stoDPeriods,movAvePeriods)
	
	#plot to check the hook and hook Price
	fig, axarr = plt.subplots(4, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].set_title(tickers[ticIdx]+'Check of Hook and Price Last Date:'+finData.date[-1].strftime('%Y-%m-%d'))
	axarr[0].plot(finData.date,finData.close,'b',label='Close')
	axarr[0].plot(finData.date,finData.high,'g',label='High')
	axarr[0].plot(finData.date,finData.low,'r',label='low')
	axarr[0].legend(loc=3,fontsize=fs)
	axarr[1].plot(finData.date,fastStok,color='k',label='fastSTO %K')
	axarr[1].plot(finData.date,stoD,color='g',label='STO %D pers:'+str(stoDPeriods))
	axarr[1].legend(loc=3,fontsize=fs)
	axarr[2].plot(finData.date,priceTechnik2,'k',label='Entry/Exit price')
	axarr[2].set_ylabel('Entry/Exit price')
	axarr[3].plot(finData.date,ordersTechnik2,'k',label='technique2 Orders')
	axarr[3].set_ylabel('technique 2 orders')
	plt.savefig(pwd+'/'+tickers[ticIdx]+'_Technique2.pdf')
	plt.close()
	
	
	latestMacdDiff=np.array(macdDiff)[-1]
	latestStok=fastStok[-1]
	latestSto=stoD[-1]
	latestMa=ma[-1]
	
	#fig , ax1 = plt.figure(facecolor='white')
	fig, axarr = plt.subplots(4, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].set_title(tickers[ticIdx]+' Last Date:'+finData.date[-1].strftime('%Y-%m-%d'))
	axarr[0].plot(finData.date,finData.close,'b',label='Close')
	axarr[0].plot(finData.date,ma,'g',label='%i day MA' % (movAvePeriods))
	axarr[0].plot(finData.date,ma20,'r',label='20 day MA')
	axarr[0].plot(finData.date,ma50,'k',label='50 day MA')
	axarr[0].plot(finData.date,ma200,'y',label='200 day MA')
	axarr[0].legend(loc=3,fontsize=fs)
	# Make the y-axis label and tick labels match the line color.
	axarr[0].set_ylabel('closing price & MAs',color='b')
	for tl in axarr[0].get_yticklabels(): tl.set_color('b')

	axarr[1].bar(finData.date,macdDiff,color='k')
	# Make the y-axis label and tick labels match the line color.
	axarr[1].set_ylabel('MACDdiff',color='k')
	for tl in axarr[1].get_yticklabels(): tl.set_color('k')
	
	axarr[2].plot(finData.date,fastStok,color='k',label='fastSTO %K')
	axarr[2].plot(finData.date,stoD,color='g',label='STO %D pers:'+str(stoDPeriods))
	# Make the y-axis label and tick labels match the line color.
	axarr[2].set_ylabel('STO%K and STO %d',color='k')
	axarr[2].legend(loc=2,fontsize=fs)
	
    
	    
	axarr[3].plot(finData.date,rule1Orders,'r')
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
	ax.plot(finData.date[-daysBack:-1],finData.close[-daysBack:-1],'b',label='Close')
	ax.plot(finData.date[-daysBack:-1],ma50[-daysBack:-1],'k',label='50 day MA')
	ax.set_title(tickers[ticIdx]+' Last Date:'+finData.date[-1].strftime('%Y-%m-%d'))
	ax.legend(loc='best',fontsize=fs) 
	ax.grid('on')
	ax.tick_params(labelsize=fs)
	ax.xaxis.label.set_fontsize(fs)
	plt.savefig(pwd+'/'+tickers[ticIdx]+'_CandlestickPast%iDays.pdf' %daysBack)
	plt.close()
	
	fig, axarr = plt.subplots(4, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].set_title(tickers[ticIdx]+' Last Date:'+finData.date[-1].strftime('%Y-%m-%d'))
	axarr[0].plot(finData.date[-daysBack:-1],finData.close[-daysBack:-1],'b',label='Close')
	axarr[0].plot(finData.date[-daysBack:-1],ma[-daysBack:-1],'g',label='%i day MA' % (movAvePeriods))
	#axarr[0].plot(finData.date[-daysBack:-1],ma20[-daysBack:-1],'r',label='20 day MA')
	axarr[0].plot(finData.date[-daysBack:-1],ma50[-daysBack:-1],'k',label='50 day MA')
	#axarr[0].plot(finData.date[-daysBack:-1],ma200[-daysBack:-1],'y',label='200 day MA')
	axarr[0].legend(loc=3,fontsize=fs)
	axarr[0].set_ylabel('closing price & MAs',color='b')
	for tl in axarr[0].get_yticklabels(): tl.set_color('b')

	axarr[1].bar(finData.date[-daysBack:-1],macdDiff[-daysBack:-1],color='k')
	# Make the y-axis label and tick labels match the line color.
	axarr[1].set_ylabel('MACDdiff',color='k')
	for tl in axarr[1].get_yticklabels(): tl.set_color('k')
	
	axarr[2].plot(finData.date[-daysBack:-1],fastStok[-daysBack:-1],color='k',label='fastSTO %K')
	axarr[2].plot(finData.date[-daysBack:-1],stoD[-daysBack:-1],color='g',label='STO %D pers:'+str(stoDPeriods))
	# Make the y-axis label and tick labels match the line color.
	axarr[2].set_ylabel('STO%K and STO %d',color='k')
	axarr[2].legend(loc=2,fontsize=fs)   
	    
	axarr[3].plot(finData.date[-daysBack:-1],rule1Orders[-daysBack:-1],'r')
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
	netWorthTek1,actionsTek1 = ff2.backTest(ordersTechnik1,finData, moneyStart,commission)	 # technique 1
	netWorthTek2,actionsTek2 = ff2.backTestWPrice(ordersTechnik2,priceTechnik2,finData, moneyStart,commission)	 # technique 2
			
	
	fig, axarr = plt.subplots(3, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].plot(finData.date,finData.close,'b',label='Close')
	axarr[0].plot(finData.date,ma,'g',label='%i day MA' % (movAvePeriods))
	axarr[0].plot(finData.date,ma20,'r',label='20 day MA')
	axarr[0].plot(finData.date,ma50,'k',label='50 day MA')
	axarr[0].plot(finData.date,ma200,'y',label='200 day MA')
	axarr[1].plot(finData.date,netWorthRule1,'k',label='netWorth Rule1')
	#axarr[1].plot(finData.date,netWorthTek1,'y',label='netWorth Technique 1')
	axarr[1].plot(finData.date,netWorthTek2,'g',label='netWorth Technique 2')
	axarr[2].plot(finData.date,actionsRule1,'ok',label='Rule1 Action taken')
	#axarr[2].plot(finData.date,actionsTek1,'*y',label='Technique 1 Action taken')
	axarr[2].plot(finData.date,actionsTek2,'sg',label='Technique 2 Action taken')
	axarr[2].plot(finData.date,rule1Orders,'k',label='Rule 1 Order')
	#axarr[2].plot(finData.date,ordersTechnik1,'y',label='Technique 1 Order')
	axarr[2].plot(finData.date,ordersTechnik2,'g',label='Technique 2 Order')
	axarr[2].set_ylim([-1.1,1.1])
	strTitle=tickers[ticIdx]+' with result Rule1=%i, Technique 1=%i, Technique 2=%i, started with %i'% (netWorthRule1[-1],netWorthTek1[-1],netWorthTek2[-1],moneyStart)
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

	