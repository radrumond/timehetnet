# Standard libraries
import os
import numpy as np
import tensorflow as tf
import gc
import numpy as np
import time
import datetime
import sys
import ast
##########
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.activations import softmax, relu
#Custom libraries
from data_sampling.data_gen_ts    import Task_Sampler
from data_sampling.load_gens      import *
###########

from time_models    import SliceEncoderModel   as SEncModel
# from informer.model import Inf_caller          as IEncModel

###########
from hetnet      import HetNet
from time_het    import TimeHetNet

###########
from experiment_test import run_experiment
from args            import argument_parser
from result_save     import *
###########

def metricSTD(y_true, y_pred):
    mse     = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    res     = mse(y_true, y_pred)
    return tf.math.reduce_std(res)

def metricMSE(y_true, y_pred):
    mse     = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    res     = mse(y_true, y_pred)
    return tf.math.reduce_mean(res)


gc.enable()
#######################################


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
print("##################################################################")


#--------Define Losses annd metrics----------------
loss_object = tf.keras.losses.MeanSquaredError()
if args.grad_clip > 0.0:
    if args.grad_clip_global:
        print(f"Using grad clip {args.grad_clip} (GLOBAL)")
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,global_clipnorm=args.grad_clip)
    else:
        print(f"Using grad clip {args.grad_clip}")
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,clipnorm=args.grad_clip)
else:
    optimizer     = tf.keras.optimizers.Adam(learning_rate=args.lr,)


#--------Define The Model----------------

model_type = "gru"
print("Building Network ...")
if args.hetmodel.lower() == "time":
    het_net    = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                            dims_pred = ast.literal_eval(args.dims_pred), 
                            activation="relu", 
                            time=args.tmax_length,
                            batchnorm=args.batchnorm, 
                            block = args.block.split(","))
    
    model_type = "TimeHetNet"
    print("Using Time Hetnet")

else:

        EncM = SEncModel(control=args.control_steps)
        het_net = HetNet(EncM, "slice",
                               dims = ast.literal_eval(args.dims),
                               acti="relu",
                               drop1=0.01,
                               drop2=0.01,
                               share_qs=False)
        print("Using Hetnet")
    

het_net.compile(loss=loss_object,optimizer=optimizer,metrics=[metricSTD])


#--------Load the data----------------
train_gen, val_gen, _, ds_names = getGens(args,model_type)       
randomnumber  = int(np.random.rand(1)*10000000)

print("-------ID Number is:", randomnumber)

#--------Define Callbacks----------------
callbacks = None
if args.early_stopping:
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=args.best_weights)]

if args.key == "debug":
    out_example = het_net.enc(next(train_gen)[0][0])
    print(out_example.numpy())
    
#------- Train the model -----------------
history = het_net.fit(              x  = train_gen,
                      validation_data  = val_gen,
                      validation_steps = 50,
                      epochs           = args.num_epochs,
                      steps_per_epoch  = 10,
                      callbacks        = callbacks)

del(train_gen)
del(val_gen)
gc.collect()


#------- Test the model -----------------
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
ts_final = run_experiment(args, het_net, ds_names,mse)



#------ Save results ---------------------
key = args.key
if key == "":
    key = None

result_path =   save_results(args        = args,
                             history     = history,
                             key         = key,
                             ts_loss     = ts_final,
                             randomnumber= randomnumber)

os.system(f"mkdir -p {result_path}_model")
if args.save_weights:
    het_net.save_weights(f"{result_path}_model/model")
