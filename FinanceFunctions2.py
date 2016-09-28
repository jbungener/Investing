#!/Users/jbungener/Python/anaconda3.2/bin/python
#!##/usr/bin/python

import numpy as np
import numpy
import pandas as pd  
import math as m
import pdb




def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi


def moving_average_convergence(x, nfast=12, nslow=26):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow,macdSmoothing):  
    EMAfast = pd.Series(pd.ewma(df['close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.ewma(MACD, span = macdSmoothing, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow)) 
    return MACD, EMAfast, EMAslow, MACDsign, MACDdiff



#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['close'], n), name = 'MA_' + str(n)) 
    return MA

#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(pd.ewma(df['close'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['close'].diff(n - 1)  
    N = df['close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(pd.rolling_mean(df['close'], n))  
    MSD = pd.Series(pd.rolling_std(df['close'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df = df.join(B2)  
    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)  
    R1 = pd.Series(2 * PP - df['low'])  
    S1 = pd.Series(2 * PP - df['high'])  
    R2 = pd.Series(PP + df['high'] - df['low'])  
    S2 = pd.Series(PP - df['high'] + df['low'])  
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))  
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

#Stochastic oscillator %K  
#def STOK(df):  
#    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')  
#     
#    return SOk

def fast_stochastic(df, period, smoothing):
	""" calculate slow stochastic
	Fast stochastic calculation
	%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
	%D = 3-day SMA of %K
	"""
	closep=df.close
	lowp=df.low
	highp=df.high
	low_min = pd.rolling_min(lowp, period)
	high_max = pd.rolling_max(highp, period)
	k_fast = 100 * (closep - low_min)/(high_max - low_min)
	nanCols=np.isnan(k_fast)
	k_fast[nanCols]=0
	d_fast = moving_average(k_fast, smoothing,'simple')
	return k_fast, d_fast


def stochastic(df, period=14, stokPeriod=1, smoothing=3):
    """ calculate slow stochastic
    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K
    """
    closep=df.close
    lowp=df.low
    highp=df.high
    k_fast, d_fast = fast_stochastic(df, period=period, smoothing=smoothing)

    # D in fast stochastic is K in slow stochastic
    k_slow = d_fast
    d_slow = moving_average(k_slow, smoothing,'simple')
    return k_fast, k_slow, d_slow


#Stochastic oscillator %D  
def STO(df, n):  
	c=df.close
	
	l, h = pd.rolling_min(c, 4), pd.rolling_max(c, 4)
	SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')  
	SOd = pd.Series(pd.ewma(SOk, span = n, min_periods = n - 1), name = 'SO%d_' + str(n)) 
	return SOk, SOd

#Trix  
def TRIX(df, n):  
    EX1 = pd.ewma(df['close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')  
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df


#Mass Index  
def MassI(df):  
    Range = df['high'] - df['low']  
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)  
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'high') - df.get_value(i, 'low')) - abs(df.get_value(i + 1, 'low') - df.get_value(i, 'high'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df





#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['close'].diff(r1 - 1)  
    N = df['close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['close'].diff(r2 - 1)  
    N = df['close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['close'].diff(r3 - 1)  
    N = df['close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['close'].diff(r4 - 1)  
    N = df['close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')  
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df

#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['Volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['high'] + df['low'] + df['close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') > 0:  
            OBV.append(df.get_value(i + 1, 'Volume'))  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') == 0:  
            OBV.append(0)  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') < 0:  
            OBV.append(-df.get_value(i + 1, 'Volume'))  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['high'] + df['low'] + df['close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['close'].diff(int(n * 11 / 10) - 1)  
    N = df['close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['close'].diff(int(n * 14 / 10) - 1)  
    N = df['close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['high'] + df['low'] + df['close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['high'] - 2 * df['low'] + df['close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['high'] + 4 * df['low'] + df['close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'close') - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['high'].ix[i:i + n - 1]) - min(df['low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['close'], n), name = 'STD_' + str(n)))  
    return df  

def buyOrSell(rData,macdFast,macdSlow,macdSmoothing,stoLength,stoDPeriods,movAvePeriods):
	# This functions evaluates the three checks that Rule#1 recommends and returns an answer whether to buy or sell or do nothing over the past data
	
	
	# Rule 1 technique
	numPeriods=np.size(rData.close)
	rule1Orders=0*np.array(range(numPeriods))# assign rule #1 orders to 0 (do nothing)
	
	MACDVal, EMAfast, EMAslow, MACDSign, MACDdiff=MACD(rData, macdFast, macdSlow, macdSmoothing)
	fastStokData, stoData=fast_stochastic(rData,stoLength,stoDPeriods)
	fastStok=np.array(fastStokData)
	stoD=np.array(stoData)
	ma =np.array(MA(rData, movAvePeriods))

	MACDBuy=MACDdiff>0 # buy if MACDdiff is >0
	stokBuy=fastStok>stoD   # buy if stochastic is above trigger
	maBuy=ma < np.array(rData.close) # buy if moving average is below the price
	
	ordBuy=MACDBuy & stokBuy & maBuy # Buy if all indices say buy
	ordSell=np.logical_not(MACDBuy) & np.logical_not(stokBuy) & np.logical_not(maBuy) # sell if all indices say sell
	#pdb.set_trace()	
	for i in range(numPeriods):
		if ordBuy[i]:
			rule1Orders[i]=1
		elif ordSell[i]:
			rule1Orders[i]=-1
			
#	Golden Cross and Silver Cross technique		
# Golden cross is buy when MA50>Ma200, sell when golden death Ma50<Ma200
# silver cross is buy when MA20>Ma50, sell when golden death Ma20<Ma50
	ma20=np.array(MA(rData, 20))
	ma50=np.array(MA(rData, 50))
	ma200=np.array(MA(rData, 200))
	goldOrders=0*np.array(range(numPeriods))# assign golden cross orders to 0 (do nothing)
	silverOrders=0*np.array(range(numPeriods))# assign silver cross orders to 0 (do nothing)
	for i in range(numPeriods):
		if ma50[i]>ma200[i]:
			goldOrders[i]=1 # gold cross says buy
		elif ma50[i]<ma200[i]:
			goldOrders[i]=-1 # gold cross says sell
		if ma20[i]>ma50[i]:
			silverOrders[i]=1 # silver cross says buy
		if ma20[i]<ma50[i]:
			silverOrders[i]=-1 # silver cross says sell
			

	return rule1Orders,goldOrders,silverOrders,np.array(MACDdiff),fastStok,stoD,ma,ma20,ma50,ma200
#############################################################

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

