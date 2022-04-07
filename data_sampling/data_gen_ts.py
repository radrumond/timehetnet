import numpy as np
import os
import random

from data_sampling.time_sampling import *


# example usage
#ts=Task_Sampler("~/data",
#                 [["ETDataset-main/ETT-small/ETTh1.csv", "ETDataset-main/ETT-small/ETTh2.csv"],
#                 ["TimeSeriesData-20211022T144600Z-001/TimeSeriesData/ECL.csv", "TimeSeriesData-20211022T144600Z-001/TimeSeriesData/WTH.csv"]])

def splitFolders(path,splitratio=0.8):
    return


class Task_Sampler():
    def __init__(self, datasets_folder,
                       splits=None):
        
        self.datasets_folder = datasets_folder
        if datasets_folder.endswith("/"):
            self.datasets_folder = datasets_folder
        else:
            self.datasets_folder = datasets_folder + "/"
        
        self.splits = splits

        self.names = { "train":       0,
                        "validation": 1,
                        "val":        1,
                        "test":       2,
                        0 : 0,
                        1 : 1,
                        2 : 2,
        }
        
        
        

        self.data = []
        for s in splits:
            data = []
            for folder in s:
                data.append(openTimeSeries(f"{os.path.expanduser(self.datasets_folder)}{folder}"))
            self.data.append(data)
        # self.test_data  = []
    def generateSet(self,
                    min_f,
                    max_f,
                    n_samples_s,
                    n_samples_q,
                    augment=False,
                    length=100,
                    max_length=None,
                    mode="train",
                    normalize=True,
                    shuffle=True,
                    better_norm=False,
                    control=False,
                    control_steps=10):
        
        
        return self.generateSetControlNormer(min_f,max_f, n_samples_s, n_samples_q, augment,length,max_length,mode,normalize, shuffle, control_steps)
        
    

    def generateSetControlNormer(self,min_f,max_f, n_samples_s, n_samples_q, augment=False,length=100,max_length=None,mode="train",normalize=True, shuffle=True, control_steps=10):
        #get the list of data-sets

        mode = self.names[mode]
        idxs = list(range(len(self.splits[mode])))
#         print (idxs)
        # check if we have a variable length
        def checkzeroPad(x):
            count   = 0
            sumation = np.sum(x,axis=(0,2))
            while count<x.shape[1] and sumation[count]==0:
                count+=1
            if count == x.shape[1]:
                count = 0
            return count

        if max_length is None:
            max_length = length
            
        while True:
            #reset task batch, and reset the feature ranges
            tasks_x = []
            tasks_y = []
            
            cur_min, cur_max = min_f, max_f
            cur_length = random.randint(length,max_length)
            n_samples = n_samples_s + n_samples_q
            
            
            if shuffle and mode!=2:
                random.shuffle(idxs)
            
            for idx in idxs:

                task_x, new_feat_size = sampleTaskControl(self.data[mode],idx,cur_length, cur_min, cur_max, n_samples,augment=False)
                
                if task_x is not None:
                    if normalize:
                        pad = checkzeroPad(task_x)
                        mean = np.mean(task_x[:,pad:,:],axis=(0,1))
                        stdx = np.std( task_x[:,pad:,:],axis=(0,1))
                        task_x[:,pad:,:] = (task_x[:,pad:,:] - mean) / (stdx+np.finfo(np.float64).eps)
                        
                    task_y = task_x[:,-1,-1:].copy()
                    if control_steps > 0:
                        task_x[:,-control_steps:,-1] = 0

                    tasks_x.append(task_x)
                    tasks_y.append(task_y)
                
                    cur_min, cur_max  = new_feat_size, new_feat_size
            # while True:

            yield     ((np.array(tasks_x)[:,:n_samples_q,:,:],
                        np.array(tasks_x)[:,n_samples_q:,:,:],
                        np.array(tasks_y)[:,n_samples_q:]),  
                        np.array(tasks_y)[:,:n_samples_q])




def get_multiv(univ_x,channel=5):
    
    tasks  = univ_x.shape[0]
    shots  = univ_x.shape[1]
    length = univ_x.shape[2]
    
    multiv_x = np.tile(univ_x,[1,1,1,channel])
    
    # Compute random breakpoints for each task to split time series
    break_points = np.random.randint(1,length,[tasks,channel])
    break_points = np.sort(break_points)
    break_points = np.concatenate([np.zeros([tasks,1]),break_points,np.ones([tasks,1])*-1],1).astype(int)
    
    # Compute 0-1 mask for meta batch
    mask = np.zeros([tasks,1,length,channel])
    for j in range(tasks):
        for i in range(channel):
            if i == channel-1:
                mask[j,0,break_points[j,i]:,i] = np.ones([len(mask[j,0])-break_points[j,i]])
            else:
                mask[j,0,break_points[j,i]:break_points[j,i+1],i] = np.ones([break_points[j,i+1]-break_points[j,i]])
    mask = np.tile(mask,[1,shots,1,1])
    
    return np.multiply(multiv_x,mask)



