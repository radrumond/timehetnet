from scipy.io import arff
import pandas as pd
import numpy as np
import os
import random
import sys

if len(sys.argv) < 3:
    print("please pass the source folder path and the destination folder path")
    exit(-1)

data_dir = sys.argv[1]

save_dir = sys.argv[2]


def prepUEA(load,target):
    print("Processing UEA datasets...")
    
    load_datasets = [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "ERing",
        "EthanolConcentration",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PEMS-SF",
        "PhonemeSpectra",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",
        "UWaveGestureLibrary"
    ]


    for fold in os.listdir(load):
        
        if fold not in load_datasets:
            continue
        else:
            print("Processing",fold)

        if not os.path.isdir(os.path.join(load,fold)):
            continue

        if not os.path.isfile(os.path.join(load,fold,fold+"_TRAIN.arff")):
            continue

        try:
            data_train = arff.loadarff(os.path.join(load,fold,fold+"_TRAIN.arff"))
            data_test  = arff.loadarff(os.path.join(load,fold,fold+"_TEST.arff"))

            df_train = pd.DataFrame(data_train[0])
            df_test = pd.DataFrame(data_test[0])
            print(fold,"loaded")
        except:
            print(fold,"crashed")
            raise
            
        df_train = pd.DataFrame(data_train[0])
        df_test = pd.DataFrame(data_test[0])
            
        x = []
        #y = []
        for sample in np.array(df_train):
            x.append(np.transpose([list(sample[0][i]) for i in range(len(sample[0]))]))

        for sample in np.array(df_test):
            x.append(np.transpose([list(sample[0][i]) for i in range(len(sample[0]))]))

        x = np.array(x)

        
        x = np.nan_to_num(x)
        
        np.save(os.path.join(target,fold+".npy"),x)

prepUEA(data_dir,save_dir)