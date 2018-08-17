#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import urllib3,requests
from datetime import timezone

def plog(text):
    print(text)


with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)['darksky']
    
with open(baseDir + '/raw/metrics.json') as f:
    metr = json.load(f)['metrics']

headers = {"Accept":"application/json","Content-type":"application/x-www-form-urlencoded; charset=UTF-8","User_Agent":"Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.6; en-US; rv:1.9.1) Gecko/20090624 Firefox/3.5"}

dayL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
dayL.loc[:,"ts"] = [datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%s") for x in dayL['day']]

BBox = metr["deBBox"]
x = (BBox[2]-BBox[0])*.5 + BBox[0]
y = (BBox[3]-BBox[1])*.5 + BBox[1]
baseUrl = "https://api.darksky.net/forecast/"
def clampF(x):
    return pd.Series({"prec":np.mean(x['precipIntensity'])
                      ,"temp_min":min(x['temperature'])
                      ,"temp_max":max(x['temperature'])
                      ,"humidity":np.mean(x['temperature'])
                      ,"pressure":np.mean(x['pressure'])
                      ,"cloudCover":np.mean(x['cloudCover'])
                      ,"visibility":np.mean(x['visibility'])
                      })

print(baseUrl + cred['token'] + "/%f,%f,%s?exclude=currently,flags" % (y,x,dayL['ts'][0]))
weathL = []
for i,c in dayL.iterrows():
    resq = requests.get(baseUrl + cred['token'] + "/%f,%f,%s?exclude=currently,flags" % (y,x,dayL['ts'][i]),headers=headers)
    locL = []
    if resq.status_code == 200:
        locL = resq.json()['daily']['data'][0]        
    locL['day'] = dayL['day'][i] 
    weathL.append(locL)

weathL = pd.DataFrame(weathL)
dayL = pd.merge(dayL,weathL,on="day",how="left")
dayL.loc[:,"ts"] = dayL['ts'].astype(int)
t = dayL['temperatureMaxTime'] - dayL['ts']
t = dayL['apparentTemperatureMax'] - dayL['apparentTemperatureHigh']
plt.hist(t,bins=20)
plt.show()
dayL.loc[:,"lightDur"] = dayL['sunsetTime'] - dayL['sunriseTime']
dayL.loc[:,"Tmin"] = (dayL['apparentTemperatureLow']-32.)*5./9.
dayL.loc[:,"Tmax"] = (dayL['apparentTemperatureHigh']-32.)*5./9.
dayL = dayL.drop(columns=['apparentTemperatureHighTime','apparentTemperatureLowTime','apparentTemperatureMax','apparentTemperatureMaxTime','apparentTemperatureMin','apparentTemperatureMinTime','cloudCoverError','moonPhase','precipAccumulation','precipIntensityMax','precipIntensityMaxTime','precipType','summary','sunriseTime','sunsetTime','temperatureHigh','temperatureHighError','temperatureHighTime','temperatureLow','temperatureLowTime','temperatureMax','temperatureMaxError','temperatureMaxTime','temperatureMin','temperatureMinTime','time','windGust','windGustTime'])
dayL = dayL.drop(columns=['apparentTemperatureLow','apparentTemperatureHigh','ts'])
dayL.replace(float('NaN'),0,inplace=True)
dayL.to_csv(baseDir + "raw/tank/dateList.csv",index=False)

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
