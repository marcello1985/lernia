import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
import findspark
findspark.init()
import pyspark
#sc = pyspark.SparkContext('local[*]')
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql import functions as func
sqlContext = SQLContext(sc)
import tarfile

def plog(text):
    print(text)

key_file = baseDir + '/credenza/geomadi.json'
cred = []
with open(key_file) as f:
    cred = json.load(f)

def parsePath(projDir,id_list=None,is_lit=False,patternN="part-00000"):
    fL = []
    for path,dirs,files in os.walk(projDir):
        for f in files:
            if re.search(patternN,f):
                fL.append(path+"/"+f)
    for i,f in enumerate(fL):
        print(f)
        print("%f" % (float(i)/float(len(fL))),end='\r',flush=True)
        try:
            df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(f)
        except:
            continue
        if id_list:
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN (" + id_list + ")")
        if is_lit:
            df = df.withColumn("dir",func.lit(i))
        if f==fL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)
    print('loaded')
    fL1 = fL
    fL1 = [re.sub(projDir,"",x) for x in fL1]
    fL1 = [re.sub("/part-00000","",x) for x in fL1]
    fL1 = [re.sub("/","-",x) for x in fL1]
    fL1 = [re.sub("^-","",x) for x in fL1]
    return ddf, fL1

def parseParquet(projDir):
    fL = []
    for path,dirs,files in os.walk(projDir):
        for f in files:
            if re.search("part-",f):
                fL.append(path+"/")
                break
    if not len(fL):
        return [],[]
    for i,f in enumerate(fL):
        print("%f" % (float(i)/float(len(fL))) )
        df = sqlContext.read.parquet(f)
        if f==fL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)
    print('loaded')
    fL1 = fL
    fL1 = [re.sub(projDir,"",x) for x in fL1]
    return ddf, fL1

def parseTar(dname,fname,idlist=None):
    print(fname)
    try : 
        tar = tarfile.open(dname+fname, "r:gz")
    except:
        return 0
    csvL = [x for x in tar.getnames() if bool(re.search("part-00000",x))]
    csvL = [x for x in csvL if bool(re.search("-201",x))]
    cs = csvL[0]
    for cs in csvL:
        t = tar.getmember(cs)
        tar.extract(cs)
        df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(cs)
        if idlist:
            df = df[df['dominant_zone'].isin(idlist)]
        act = df.toPandas()
        os.remove(cs)
        dName = cs.split("/")[0][31:]
        #dName = fname.split(".")[0]
        act.dropna(inplace=True)
        act.loc[:,"time"] = act['time'].apply(lambda x: x[:8])
        try:
            act.to_csv(baseDir + projDir + dName + ".csv",index=False)
        except:
            act.to_csv(baseDir + "out/" + dName + ".csv",index=False)            

    tar.close()
    return act
