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
#import FinanceFunctions as ff
import FinanceFunctions2 as ff2
import os


#######################################################################################
def backTest(orders,finData, moneyStart,commission): # run a simulation using the available data. 

	money=moneyStart# Dollars to start with
	nShares=0 # we start with 0 shares
	actions=[0 for i in range (np.size(orders))]
	netWorth=[0 for i in range (np.size(orders))]
	
	for dayIdx in range(np.size(orders)-1): # skip the last day b/c we would sell the last day anyways to get the Money amount. 
		netWorth[dayIdx]=money+nShares*finData.open[dayIdx+1]
		#start trading
		if orders[dayIdx]==0: continue
		elif orders[dayIdx]==-1 and nShares==0: continue # if sell and we have not bouhgt anything then do nothing
		elif orders[dayIdx]==1 and nShares>0: continue # If order is to buy and we already have bought.
		elif orders[dayIdx]==-1 and nShares>0: # need to sell
			actions[dayIdx]=-1
			money=money+nShares*finData.open[dayIdx+1]-commission # we would sell the next morning
			nShares=0	
		elif orders[dayIdx]==1 and nShares==0: # need to buy
			actions[dayIdx]=1
			nShares= int(money/finData.open[dayIdx+1]) # int acts as a floor. So e buy the number of shares we can with the money we have. 
			money=money-nShares*finData.open[dayIdx+1]-commission
		else:
			exit('Error in the if else logic for buying and selling')
	#Last day:
	if nShares>0:	money=money+nShares*finData.open[dayIdx+1]-commission
	netWorth[-1]=money
	return netWorth,actions

#######################################################################################

pwd=os.getcwd()

startdate = datetime.date(2015, 5, 1)# Date I started
today = enddate = datetime.date.today()
tickers = ['NGVC','CMG','BRK-B','ATRO','BIDU','ANIK','WETF','FRAN','RHT','LL','EGOV']


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

	fh = finance.fetch_historical_yahoo(tickers[ticIdx], startdate, enddate)
# a numpy record array with fields: date, open, high, low, close, volume, adj_close)
	finData = mlab.csv2rec(fh)
	fh.close()
	finData.sort()

	timeDelt=finData.date[-1]-finData.date[1]
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
	axarr[0].set_title(tickers[ticIdx])
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
	
	
	#Now we look at the performance in the past X many days
	moneyStart=10000# Dollars to start with
	commission=6.99 # transaction dollars	
	
	netWorthRule1,actionsRule1 = backTest(rule1Orders,finData, moneyStart,commission)	# Rule 1
	netWorthGC,actionsGC = backTest(goldOrders,finData, moneyStart,commission)	 # Gold Cross
	netWorthSC,actionsSC = backTest(silverOrders,finData, moneyStart,commission)	 # Silver Cross
			
	
	fig, axarr = plt.subplots(3, sharex=True,figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
	axarr[0].plot(finData.date,finData.close,'b',label='Close')
	axarr[0].plot(finData.date,ma,'g',label='%i day MA' % (movAvePeriods))
	axarr[0].plot(finData.date,ma20,'r',label='20 day MA')
	axarr[0].plot(finData.date,ma50,'k',label='50 day MA')
	axarr[0].plot(finData.date,ma200,'y',label='200 day MA')
	axarr[1].plot(finData.date,netWorthRule1,'k',label='netWorth Rule1')
	axarr[1].plot(finData.date,netWorthGC,'y',label='netWorth Gold Cross')
	axarr[1].plot(finData.date,netWorthSC,'g',label='netWorth Silver Cross')
	axarr[2].plot(finData.date,actionsRule1,'ok',label='Rule1 Action taken')
	axarr[2].plot(finData.date,actionsGC,'*y',label='Gold Cross Action taken')
	axarr[2].plot(finData.date,actionsSC,'sg',label='Silver Cross Action taken')
	axarr[2].plot(finData.date,rule1Orders,'k',label='Rule 1 Order')
	axarr[2].plot(finData.date,goldOrders,'y',label='Gold Cross Order')
	axarr[2].plot(finData.date,silverOrders,'g',label='Silver Cross Order')
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

	