# TimeHetNet Meta-dataset
In order to reproduce the results from our paper, you must rebuild our training data-set and our test set. Please acquire, pre-process and place the files as specified below. Please make sure the folder structure is THE SAME since we hardcoded these paths into the fold splits files.

## PeekDB
The peekdb data-set is available [here](https://github.com/RafaelDrumond/PeekDB/tree/master/TimeHetNet). And it's already pre-processed.

## Informer data-sets
The informer data-sets are available [here](https://github.com/zhouhaoyi/Informer2020) and require no pre-processing.

## Monash
1. Download the respective monash datasets in .tsf format from [here](https://zenodo.org/communities/forecasting) and place them in ```<folder_path_with_tsf_files>```
    - The electricity and traffics dataset are both the hourly version
    
2. Download the monash ```data_loader.py``` from the offical repository and place it in timehetnet/metadataset: [link](https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py)

3. Process all monash datasets by running the ```monash.py``` with
```bash
python monash.py <folder_path_with_tsf_files> <destination_folder>
```

## UEA
1. Download the multivariate UEA Time Series Classification datasets in .arff format from [here](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip) and place them in ```<folder_path_with_arff_files>```
3. Process the datasets by running the ```uea.py``` with
```bash
python uea.py <folder_path_with_arff_files> <destination_folder>
```


## Kaggle challenge datasets
### CNC

1. First download the "experiment" csv files from [here](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill/download).
2. Place all the csvs on a folder with only the experiment*.csv files.
3. Run the ```kaggle.py``` with
```bash
python kaggle.py <folder_path_with_csv_files> <destination_folder> cnc
```

### Mining

1. First download the csv file from [here](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process/download).
2. Place the csvs on a folder with only the csv file.
3. Run the ```kaggle.py``` with
```bash
python kaggle.py <folder_path_with_csv_files> <destination_folder> mining
```

### Plant

1. First download the csv files from [here](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring/download).
2. Place the csvs on a folder with only the csv files.
3. Run the ```kaggle.py``` with
```bash
python kaggle.py <folder_path_with_csv_files> <destination_folder> plant
```



## Data folder structure

Make sure the data folder looks like this:

```
     #PEEK
     data/peekdb/x.pkl.npy

     #Informer
     data/ETDataset-main/ETT-small/ETTh1.csv
     data/ETDataset-main/ETT-small/ETTh2.csv
     data/ETDataset-main/ETT-small/ETTm1.csv
     data/ETDataset-main/ETT-small/ETTm2.csv
     data/TimeSeriesData-20211022T144600Z-001/TimeSeriesData/ECL.csv
     data/TimeSeriesData-20211022T144600Z-001/TimeSeriesData/WTH.csv

     #KAGGLE
     data/cnc/cnc.pkl.npy
     data/mining/mining.npy
     data/plant_monitoring/plant.pkl.npy
     
     #MONASH
     data/Monash/Covid.npy
     data/Monash/Electricity.npy
     data/Monash/FRED.npy
     data/Monash/Rideshare.npy]
     data/Monash/Temperature.npy
     data/Monash/Traffic.npy
     
     #UEA
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
python generate_test_set.py
```

you can also specify your data directory with ``--data_dir``, the default is `~/data`. If you use a different data directory, make sure to specify it when running `experiment.py` as well.
