# Standard libraries
import os
import numpy as np
# import tensorflow as tf
import gc
import numpy as np
import time
import datetime
import sys

import ast


from data_sampling.data_gen_ts    import Task_Sampler
from data_sampling.load_gens      import *

from args import argument_parser
from result_save import *
import os,sys
import gc
gc.enable()

import random



random.seed(16)
np.random.seed(16)

args = argument_parser()
file_time = str(datetime.datetime.now()).replace(" ","_")
ft = "file_time"
if args.name is None:
    name = file_time
else:
    name = args.name+"_"+file_time

args.name = name

print("########## argument sheet ########################################")
for arg in vars(args):
    print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
    #print(f"#{ft:>15}  :  {file_time} ")
print("##################################################################")
model_type = ""
args.control_mode = True
args.control_steps = 0
args.better_norm = True
fs = [5,6,7,8,9,10]
for sps in [0,1,2,3,4]:
    args.split = sps
    for f in fs:
        args.min_f, args.max_f = f, f
        print("split: ",args.split,args.min_f, args.max_f)
        _, _, test_gen, _ = getGens(args,model_type,load_test=True,onlyTest=True)       

        tasks = []

        for _ in range (1000):
            ((qx,sx,sy),qy) = next(test_gen)
            qx = np.float32(qx)
            sx = np.float32(sx)
            sy = np.float32(sy)
            qy = np.float32(qy)
            n = ((qx,sx,sy),qy)
            tasks.append(n)

        os.system(f"mkdir -p ./splits/")
        os.system(f"mkdir -p ./splits/{args.split}_control_test_better")
        np.save(f"splits/{args.split}_control_test_better/control_test{f}.npy", np.array(tasks))
        del(test_gen)
        del(tasks)
        gc.collect()

