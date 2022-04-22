import numpy as np
import os
import random
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("please pass the source folder path and the destination folder path")
    exit(-1)

data_dir = sys.argv[1]

save_dir = sys.argv[2]

try:
    from monash_data_loader import convert_tsf_to_dataframe
except:
    print("Please download the monash data loader from the official git repo: https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py")
    raise
    
    
datasets = ["covid_deaths_dataset.tsf",
            "electricity_hourly_dataset.tsf",
            "fred_md_dataset.tsf",
            "traffic_hourly_dataset.tsf",
            "temperature_rain_dataset_without_missing_values.tsf",
            "rideshare_dataset_without_missing_values.tsf"]

for ds in datasets:
    cur_path = os.path.join(data_dir,ds)
    
    if not os.path.exists(cur_path):
        print(f"Please download the .tsf data for {ds} from the official monash repo: https://zenodo.org/communities/forecasting")
    
    loaded_data,frequency,forecast_horizon,contain_missing_values,contain_equal_length = convert_tsf_to_dataframe(cur_path)
    
    if "rideshare" in ds:
        
        ### Processing rideshare ###
        
        # all attributes in the dataset
        columns_all = np.array(['price_min', 'price_mean', 'price_max', 'distance_min',
               'distance_mean', 'distance_max', 'surge_min', 'surge_mean',
               'surge_max', 'api_calls', 'temp', 'rain', 'humidity', 'clouds',
               'wind'])

        # group samples by source_location-provider_name-provider_service
        arr = np.array(loaded_data)
        data = np.expand_dims(np.array([arr[i,1]+arr[i,2]+arr[i,3] for i in range(len(arr))]),-1)
        data = np.concatenate([data,arr[:,4:5],arr[:,6:7]],axis=-1)
        samples = np.unique(data[:,0])

        # Collect all attributes for all samples for each group of source_location-provider_name-provider_service
        dataset = []
        for sample in samples:
            subset = np.where(data[:,0]==sample)[0]
            columns = data[subset]
            columns = columns[:,1]

            df_data = data[subset]
            df_data = np.array([np.array(i) for i in df_data[:,-1]]).transpose()

            df = pd.DataFrame(df_data)
            df.columns = columns

            if len(columns) < len(columns_all):
                for c in columns_all:
                    if c not in columns:
                        df[c] = np.empty(len(df))*np.nan

            df = df[columns_all]
            dataset.append(np.array(df))
        dataset = np.array(dataset)
        
    elif "temperature" in ds:
        
        # Temperature Rain
        data = np.array(loaded_data)
        stations = np.unique(data[:,1])
        dataset = []
        for idx,s in enumerate(stations):
            if idx%100==0:
                print(idx,"/",len(stations))
            subset = np.where(np.array(loaded_data)[:,1]==s)[0]

            columns = data[subset]
            columns = columns[:,2]

            df_data = data[subset]
            df_data = np.array([np.array(i) for i in df_data[:,4]]).transpose()

            dataset.append(np.array(df_data))
        dataset = np.array(dataset)
        
    else:

        dataset = np.array([np.array(i) for i in np.array(loaded_data)[:,-1]])
        dataset = np.transpose(dataset)
        
        
    print(ds,dataset.shape)
    np.save(os.path.join(save_dir,ds.split("_")[0]),dataset)