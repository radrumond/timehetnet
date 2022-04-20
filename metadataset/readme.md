# TimeHetNet Meta-dataset
In order to reproduce the results from our paper, you must rebuild our training data-set and our test set.

## PeekDB
The peekdb data-set is available [here](https://github.com/RafaelDrumond/PeekDB/tree/master/TimeHetNet). And it's already pre-processed.

## Informer data-sets
The informer data-sets are available [here](https://github.com/zhouhaoyi/Informer2020) and require no pre-processing.

## Monash
Coming soon...

## UEA
Coming soon...

## CNC, Plant_Monitoring and Mining (Kaggle challenge data-sets)
Coming soon...


## Data folder structure

Make sure the data folder looks like this:

```
     data/peekdb/x.pkl.npy

     data/ETDataset-main/ETT-small/ETTh1.csv
     data/ETDataset-main/ETT-small/ETTh2.csv
     data/ETDataset-main/ETT-small/ETTm1.csv
     data/ETDataset-main/ETT-small/ETTm2.csv
     data/TimeSeriesData-20211022T144600Z-001/TimeSeriesData/ECL.csv
     data/TimeSeriesData-20211022T144600Z-001/TimeSeriesData/WTH.csv

     data/cnc/cnc.pkl.npy
     data/mining/mining.npy
     data/plant_monitoring/plant.pkl.npy

     data/Monash/Covid.npy
     data/Monash/Electricity.npy
     data/Monash/FRED.npy
     data/Monash/Rideshare.npy]
     data/Monash/Temperature.npy
     data/Monash/Traffic.npy

     data/UEA_multivariate/x/ArticularyWordRecognition_x.npy
     data/UEA_multivariate/x/AtrialFibrillation_x.npy
     data/UEA_multivariate/x/BasicMotions_x.npy
     data/UEA_multivariate/x/Cricket_x.npy
     data/UEA_multivariate/x/DuckDuckGeese_x.npy
     data/UEA_multivariate/x/EigenWorms_x.npy
     data/UEA_multivariate/x/Epilepsy_x.npy
     data/UEA_multivariate/x/ERing_x.npy
     data/UEA_multivariate/x/EthanolConcentration_x.npy
     data/UEA_multivariate/x/FaceDetection_x.npy
     data/UEA_multivariate/x/FingerMovements_x.npy
     data/UEA_multivariate/x/HandMovementDirection_x.npy
     data/UEA_multivariate/x/JapaneseVowels_x.npy
     data/UEA_multivariate/x/Libras_x.npy
     data/UEA_multivariate/x/LSST_x.npy
     data/UEA_multivariate/x/MotorImagery_x.npy
     data/UEA_multivariate/x/NATOPS_x.npy
     data/UEA_multivariate/x/PEMS-SF_x.npy
     data/UEA_multivariate/x/PhonemeSpectra_x.npy
     data/UEA_multivariate/x/SelfRegulationSCP1_x.npy
     data/UEA_multivariate/x/SelfRegulationSCP2_x.npy
     data/UEA_multivariate/x/SpokenArabicDigits_x.npy
     data/UEA_multivariate/x/StandWalkJump_x.npy
     data/UEA_multivariate/x/UWaveGestureLibrary_x.npy
```

## Building the test set
From the root folder of this repo, run
```bash
python generate_test_sets/generate_test_set.py
```

you can also specify your data directory with ``--data_dir``, the default is `~/data`. If you use a different data directory, make sure to specify it when running `experiment.py` as well.
