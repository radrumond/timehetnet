import numpy as np
import os
import random


# Functions for opening data-set files
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
                time_stamp = parseTime(line[0])
                for l in line[1:]:
                    time_stamp.append(float(l))
                data.append(time_stamp)
            data = np.array(data)
        return np.nan_to_num(data)
    elif path.endswith(".pkl.npy"):
        return np.nan_to_num(np.load(path,allow_pickle=True))
    elif path.endswith(".npy"):
        return np.nan_to_num(np.load(path))
    else:
        return [np.nan_to_num(np.load(os.path.join(path,i))) for i in  os.listdir(path)]

# feature and task sampling functions

def featSample(requested, total_feats):
    count = 1

        
    samps = []
    
    range_total_f = list(range(total_feats))
    random.shuffle(range_total_f)
    label = range_total_f[0]
    feats = range_total_f[1:]
    while count < requested:
        if requested - count > len(feats):
            samps = samps + random.sample(feats, len(feats))
            count+= len(feats)
        else:
            samps = samps + random.sample(feats, requested - count)
            count = requested
    random.shuffle(samps)

    samps.append(label)
    return samps


def sampleTime(datasets,idx,length,min_f,max_f,n_samples):
    
    cur_ds  = datasets[idx]
    minibatch=[]
    
    if len(cur_ds.shape) == 2:
        
        max_len = len(cur_ds)
        total_f = cur_ds.shape[-1]

        #     feats   = min(max_f,total_f)
        #     if not(min_f == max_f):
        #         feats   = random.randint(min_f,min(max_f,total_f))
        
        feats   = random.randint(min_f,max_f)
        feats   = featSample(feats, total_f) #random.sample(range(total_f), feats)
        
        
        for i in random.sample(range(max_len-length-1), min(n_samples,max_len-length)):
            minibatch.append(cur_ds[i:i+length,feats])

        return np.array(minibatch),len(feats)
    
    else:
        max_len = len(cur_ds)
        total_f = cur_ds[0].shape[-1]
        feats   = random.randint(min_f,max_f)
        feats   = featSample(feats, total_f)
        
        ###############################################
        # MAke sure samples from query and support do not intersect
        ##############################################
        total_samps = list(range(max_len))
        random.shuffle(total_samps)
        q_s_list = total_samps[:int(max_len/2)]
        s_s_list = total_samps[int(max_len/2):]

        replace = False
        if n_samples/2 > len(q_s_list):
            replace=True

        q_s_list = np.random.choice(q_s_list, size=int(n_samples/2), replace=replace)
        s_s_list = np.random.choice(s_s_list, size=int(n_samples/2), replace=replace)

        choice_list = np.concatenate([q_s_list,s_s_list])
        #################################################
        
        for i in choice_list:
            # print("woot",cur_ds.shape,length) 
            if cur_ds[i].shape[0]-length >= 0:
                upper_bound  = cur_ds[i].shape[0]-length
                start_idx    = random.randint(0,upper_bound)
                cur_sample = cur_ds[i][start_idx:start_idx+length,feats]
                # assert cur_sample.shape == (100,5),f"Wrong shape top ({cur_sample.shape}) after sampling from ds with shape {cur_ds.shape}, start_idx,length: {(start_idx,length)}"
                minibatch.append(cur_sample)
            else:

                fill_quota = length - cur_ds[i].shape[0]
                # l= random.randint(0,fillquota)
                # zeros_l = np.zeros([,len(feats)])
                # if l > 0
                zeros_r = np.zeros([fill_quota,len(feats)])

                cur_sample = cur_ds[i][:,feats]

                cur_sample = np.concatenate([zeros_r,cur_sample],axis=0)
         
                minibatch.append(cur_sample)

                
        
        return np.array(minibatch), len(feats)


def augmentTS(samples):
    return samples

def sampleTaskControl(datasets,idx,length,min_f,max_f,n_samples,augment=False):
    zero_flag = True
    std_flag  = True
    std_lives = 5
    while zero_flag or (std_flag and std_lives > 0):

        mini_batch, feats = sampleTime(datasets,idx,length,min_f,max_f,n_samples)
        if feats is None:
            return None, None, None
        if np.sum(np.abs(mini_batch)) != 0.0:
            zero_flag = False
            
        tmp = np.mean(mini_batch[:,1:,:]==mini_batch[:,:-1,:],axis=(0,1))
        if max(tmp) > 0.95:
            std_lives = std_lives - 1
        else:
            std_flag = False
            
            
    
    mini_batch  = augmentTS(mini_batch)
    x_batch = mini_batch[:,:,:] 
    #, mini_batch[:,-1,channels]
    return x_batch,feats