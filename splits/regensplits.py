import numpy as np

def txtsave(i,txt):
    with open(f"{i}.txt", "w+") as f:
        f.write(txt)

def savetxts(splits_new): 
    for i,s in enumerate(splits_new):
        txt = ""
        for sx in s:
            for ss in sx:
                if txt == "":
                    txt = txt + f"{ss}"
                else:
                    txt = txt + f"\n{ss}"
            txt = txt + f"\n-"
        txtsave(i,txt)
        
sps = [
    
       ['UEA_multivariate/x/HandMovementDirection_x.npy',
        'UEA_multivariate/x/SelfRegulationSCP1_x.npy',
        'UEA_multivariate/x/Epilepsy_x.npy',
        'ETDataset-main/ETT-small/ETTm1.csv',
        'UEA_multivariate/x/UWaveGestureLibrary_x.npy',
        'ETDataset-main/ETT-small/ETTm2.csv',
        'UEA_multivariate/x/NATOPS_x.npy',
        'UEA_multivariate/x/LSST_x.npy'],
    
       ['UEA_multivariate/x/EigenWorms_x.npy',
        'UEA_multivariate/x/PEMS-SF_x.npy',
        'TimeSeriesData-20211022T144600Z-001/TimeSeriesData/ECL.csv',
        'UEA_multivariate/x/BasicMotions_x.npy',
        'ETDataset-main/ETT-small/ETTh2.csv',
        'UEA_multivariate/x/ArticularyWordRecognition_x.npy',
        'UEA_multivariate/x/FingerMovements_x.npy',
        'Monash/Rideshare.npy'],
    
       ['UEA_multivariate/x/DuckDuckGeese_x.npy',
        'UEA_multivariate/x/PhonemeSpectra_x.npy',
         'Monash/FRED.npy',
        'UEA_multivariate/x/AtrialFibrillation_x.npy',
        'UEA_multivariate/x/StandWalkJump_x.npy',
        'UEA_multivariate/x/FaceDetection_x.npy', 
        'cnc/cnc.pkl.npy',
        'Monash/Covid.npy'],
    
       ['UEA_multivariate/x/EthanolConcentration_x.npy',
        'Monash/Traffic.npy', 
        'Monash/Electricity.npy',
        'peekdb/x.pkl.npy', 
        'UEA_multivariate/x/Cricket_x.npy',
        'TimeSeriesData-20211022T144600Z-001/TimeSeriesData/WTH.csv',
        'UEA_multivariate/x/SpokenArabicDigits_x.npy',
        'UEA_multivariate/x/ERing_x.npy'],
    
       ['UEA_multivariate/x/SelfRegulationSCP2_x.npy',
        'UEA_multivariate/x/MotorImagery_x.npy',
        'ETDataset-main/ETT-small/ETTh1.csv', 
        'mining/mining.npy',
        'plant_monitoring/plant.pkl.npy', 
        'Monash/Temperature.npy',
        'UEA_multivariate/x/Libras_x.npy',
        'UEA_multivariate/x/JapaneseVowels_x.npy']
]

splits_new = [
    [sps[1]+sps[2]+sps[3],sps[4],sps[0]],
    [sps[2]+sps[3]+sps[4],sps[0],sps[1]],
    [sps[3]+sps[4]+sps[0],sps[1],sps[2]],
    [sps[4]+sps[0]+sps[1],sps[2],sps[3]],
    [sps[0]+sps[1]+sps[2],sps[3],sps[4]]    
]

np.save("splits.npy",splits_new)
savetxts(splits_new)


