import numpy as np
import os
import random
import sys

def parseTime(date):
    time_feats = []
    return time_feats

def openTimeSeries(path):
    if path.endswith(".csv"):
        with open(path) as file:
            lines = file.readlines()
            data = []
            for line in lines[1:]:
                line = line.split(",")
                time_stamp = []
                for l in line[1:]:
                    if is_float(l):
                        time_stamp.append(float(l))
                if len(time_stamp)>0:
                    data.append(np.array(time_stamp))
                
            data = np.array(data)
        return data
    return None
def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
    
if len(sys.argv) < 3:
    print("please pass the source folder path and the destination folder path")
    exit(-1)

direct = sys.argv[1]
if not direct.endswith('/'):
    direct+='/'

dest = sys.argv[2]
if not dest.endswith('/'):
    dest+='/'
    
    
files = []

if sys.argv[3] == "mining":
    x = openTimeSeries(direct+"MiningProcess_Flotation_Plant_Database.csv")
    print(x.shape)
    np.save(dest+"mining.npy",x)
elif sys.argv[3] == "cnc" or sys.argv[3] == "plant":
    for i in os.listdir(direct):
        if i.endswith("csv"):
            print(i,direct+i)
            x = None
            x = openTimeSeries(direct+i)
            print(x.shape)
            if x is not None:
                files.append(x)
            
    if sys.argv[3] == "cnc":
        np.save(dest+"cnc.pkl.npy",files)
    elif sys.argv[3] == "plant":
        np.save(dest+"plant.pkl.npy",files)
        
else:
    print("error, wrong data-set")
