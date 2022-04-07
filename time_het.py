import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax, relu
#from models.chameleon import *
import os
import numpy as np
import time
tf.keras.backend.set_floatx('float32')

class TimeHetNet(tf.keras.Model):

    def __init__(self,
                dims_inf = [32,32,32],
                dims_pred = [32,32,32],
                activation = "relu",
                time=100,
                batchnorm = False,
                block = "conv",
                merge = False,
                dropout = 0.0):
        
        super(TimeHetNet, self).__init__()
        self.enc_type  = "None"

        if len(block) == 1:
            block = f"{block},{block},{block},{block}"

        self.block = block

        # # Prediction network
        self.dense_fz = getSequential(dims=dims_pred,activation=activation,final=True,name="pred_dense_fz")


        self.time_fz = getTimeBlock(block=block[-1],dims=dims_pred,activation=activation,final=False,name="pred_time_fz",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)

        self.time_gz = getTimeBlock(block=block[-1],dims=dims_pred,activation=activation,final=False,name="pred_time_gz",input_shape=(time,dims_pred[-1]),batchnorm=batchnorm)


        # # Support and Query network (start with both same weights)
        self.time_fv = getTimeBlock(block=block[2],dims=dims_inf,activation=activation,final=False,name="s_time_fv",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)
        self.time_gv = getTimeBlock(block=block[2],dims=dims_inf,activation=activation,final=False,name="s_time_gv",input_shape=(time,dims_inf[-1]),batchnorm=batchnorm)
        self.dense_fv = getSequential(dims=dims_inf,activation=activation,final=False,name="s_dense_fv")


        # # U net
        self.dense_uf = getSequential(dims=dims_inf,activation=activation,final=False,name="ux_dense_f")
        self.time_uf = getTimeBlock(block=block[1],dims=dims_inf,activation=activation,final=False,name="ux_time_f",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)
        self.time_ug = getTimeBlock(block=block[1],dims=dims_inf,activation=activation,final=False,name="ux_time_g",input_shape=(time,dims_inf[-1]),batchnorm=batchnorm)

        # # Vbar network
        self.time_v  = getTimeBlock(block=block[0],dims=dims_inf,activation=activation,final=False,name="vb_time_v",input_shape=(time,1),batchnorm=batchnorm)
        self.time_c  = getTimeBlock(block=block[0],dims=dims_inf,activation=activation,final=False,name="vb_time_c", input_shape=(time,dims_inf[-1]),batchnorm=batchnorm)

        self.dense_v  = getSequential(dims=dims_inf,activation=activation,final=False,name="vb_dense_v")
        self.dense_c  = getSequential(dims=dims_inf,activation=activation,final=False,name="vb_dense_c")

        

    # input should be [Metabatch x samples X Time X features] and [Metabatch samples X labels]
    def call(self, inp, training=False):
        que_x, sup_x, sup_y = inp

        M = tf.shape(sup_x)[0] # Metabatch
        N = tf.shape(sup_x)[1] # Batch
        T = tf.shape(sup_x)[2] # Time
        F = tf.shape(sup_x)[3] # Channels/Features


        zero_count = tf.reduce_sum(tf.concat([que_x,sup_x],axis=1),axis=[1,3])
        zero_count = tf.math.count_nonzero(zero_count,axis=1,dtype=tf.dtypes.float32)
        zero_count = tf.expand_dims(zero_count,-1)
        zero_count = tf.expand_dims(zero_count,-1)
        
        ##### Vbar network #####

        # Encode sup_x MxNxTxF to MxFxTxK (DS over Instances)
        vs_bar = tf.transpose(sup_x,[0,1,3,2])  # MxNxFxT
        vs_bar = tf.expand_dims(vs_bar,-1)      # MxNxFxTx1
        vs_bar = self.time_v(vs_bar,training)            # MxNxFxTxK

        vs_bar = tf.reduce_mean(vs_bar, axis=1) # MxFxTxK
        vs_bar = self.time_c(vs_bar,training) # MxFxTxK
        vs_bar = tf.transpose(vs_bar,[0,2,1,3]) # MxTxFxK

        # Encode sup_y MxNx1 to Mx1xK
        cs_bar = tf.expand_dims(sup_y,axis=-1)  # MxNx1x1    
        cs_bar = self.dense_v(cs_bar)            # MxNx1xK 
        cs_bar = tf.reduce_mean(cs_bar, axis=1) # Mx1xK 
        cs_bar = self.dense_c(cs_bar)           # Mx1xK 
        

        ##### U network #####  (DS over Channels)

        vs_bar  = tf.tile(tf.expand_dims(vs_bar,axis=1),[1,N,1,1,1]) # MxNxTxFxK

        sup_x_1 = tf.expand_dims(sup_x,axis=-1) # MxNxTxFx1
        u_xs = tf.concat([sup_x_1,vs_bar],-1) # MxNxTxFx(K+1)

        u_xs = tf.transpose(u_xs,[0,1,3,2,4]) # MxNxFxTx(K+1)
        u_xs = self.time_uf(u_xs,training) # MxNxFxTxK


        u_xs = tf.reduce_mean(u_xs, axis=2) # MxNxTxK


        cs_bar = tf.tile(tf.expand_dims(cs_bar,axis=1),[1,N,1,1]) # MxNx1xK 
        u_ys = tf.concat([tf.expand_dims(sup_y,axis=-1),cs_bar],axis=-1)                      # MxNx1x(K+1) 
        u_ys = self.dense_uf(u_ys)          # MxNx1xK
        u_ys = tf.reduce_mean(u_ys, axis=2) # MxNxK
        u_ys = tf.expand_dims(u_ys,2)       # MxNxK
        u_ys = tf.tile(u_ys,[1,1,T,1])      # MxNxTxK


        u_s  = u_xs + u_ys # MxNxTxK  

        u_s = self.time_ug(u_s,training)    # MxNxTxK

        #### Inference Network #### (DS over Instances)
        in_xs = tf.tile(tf.expand_dims(u_s,axis=3),[1,1,1,F,1]) # MxNxTxFxK
        in_xs = tf.concat([sup_x_1,in_xs],-1) # MxNxTxFx(K+1)

        # Label encoding; useful?
        in_ys = tf.reduce_mean(u_s,axis=2)    # MxNxK

        in_ys = tf.concat([sup_y,in_ys],axis=-1) # MxNx(K+1)

        in_ys = self.dense_fv(in_ys)        # MxNxK


        in_xs = tf.transpose(in_xs,[0,1,3,2,4]) # MxNxFxTx(K+1)
        in_xs = self.time_fv(in_xs,training) # MxNxFxTxK
        in_xs = tf.reduce_mean(in_xs, axis=1) # MxFxTxK
        in_xs = self.time_gv(in_xs,training)     # MxFxTxK
        in_xs = tf.transpose(in_xs,[0,2,1,3]) # MxTxFxK

        #### Prediction Network ####
        p_xs = tf.tile(tf.expand_dims(in_xs, axis=1),[1,N,1,1,1]) # MxNxTxFxK
        que_x_1 = tf.expand_dims(que_x, axis=-1) # MxNxTxFx1

        

        z = tf.concat([p_xs,que_x_1],axis=-1) # MxNxTxFx(K+1)
            
        z = tf.transpose(z,[0,1,3,2,4]) # MxNxFxTx(K+1)

        z = self.time_fz(z,training) # MxNxFxTxK

        # (Ds over Channels)
        z = tf.reduce_mean(z, axis=2) # MxNxTxK

        
        z = self.time_gz(z,training) # MxNxTxK


        if self.block[-1] == "gru":
            out = z[:,:,-1,:]
        else:
            # reduce time away
            if self.zero_div:
                out = tf.reduce_sum(z,axis=-2)
                out = tf.math.divide_no_nan(out,zero_count)
            else:
                out = tf.reduce_mean(z,axis=2) # MxNxK

        out = tf.concat([out,in_ys],-1) # MxNx(K+K)
        out = self.dense_fz(out)

        return out

def getTimeBlock(block = "conv", dims=[32,32,1],input_shape=None,activation=None,name=None,final=True,batchnorm=False,dilate=False):

    if block == "conv":

        return convBlock(dims=dims,input_shape=input_shape,activation=activation,name=name,final=final,batchnorm=batchnorm,dilate=dilate)

    elif block == "gru":

        return gruBlock(dims=dims,input_shape=input_shape,activation=activation,name=name,final=final)

    else:
        raise ValueError(f"Block type {block} not defined.")


class convBlock(tf.keras.Model):

    def __init__(self,dims=[32,32,1],input_shape=None,activation=None,name=None,final=True,batchnorm=False,dilate=False):
        
        super(convBlock, self).__init__()

        self.batchnorm = batchnorm
        self.final = final
        dilation = [1,1,1]
        
        self.c1 = tf.keras.layers.Conv1D(filters=dims[0],kernel_size=3, activation=None,name=f"{name}-0",padding="same",dilation_rate=dilation[0],input_shape=input_shape)
        self.relu1 = tf.keras.layers.Activation(activation)

        self.c2 = tf.keras.layers.Conv1D(filters=dims[1],kernel_size=3, activation=None,name=f"{name}-1",padding="same",dilation_rate=dilation[1],input_shape=(input_shape[0],dims[0]))
        self.relu2 = tf.keras.layers.Activation(activation)


        self.c3 = tf.keras.layers.Conv1D(filters=dims[2],kernel_size=3, activation=None,name=f"{name}-2",padding="same",dilation_rate=dilation[2],input_shape=(input_shape[0],dims[1]))  
        if not self.final:
            self.relu3 = tf.keras.layers.Activation(activation) 

        if self.batchnorm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()                 
        

    def call(self, inp, training=False):
        
        out = self.c1(inp)
        if self.batchnorm:
            out = self.bn1(out,training)
        out = self.relu1(out)

        out = self.c2(out)
        if self.batchnorm:
            out = self.bn2(out,training)
        out = self.relu2(out)

        out = self.c3(out)
        if self.batchnorm:
            out = self.bn3(out,training)
        if not self.final:
            out = self.relu3(out)

        return out


class gruBlock(tf.keras.Model):

    def __init__(self,dims=[32,32,1],input_shape=None,activation=None,name=None,final=False):
        
        super(gruBlock, self).__init__()

        self.final = final

        self.g1 = tf.keras.layers.GRU(units=dims[0], return_sequences=True, return_state=True,name=f"{name}-0",input_shape=input_shape)
        self.g2 = tf.keras.layers.GRU(units=dims[1], return_sequences=True, return_state=True,name=f"{name}-1",input_shape=input_shape)
        self.g3 = tf.keras.layers.GRU(units=dims[2], return_sequences=True, return_state=True,name=f"{name}-3",input_shape=input_shape)              

    def call(self, inp, training=False):
        #input is TASKS x SAMPLES x TIME X FEATURES
        shape = tf.shape(inp)
        x = tf.reshape(inp,[-1,shape[-2],shape[-1]])
        
        x,f = self.g1(x)
        x,f = self.g2(x)
        x,f = self.g3(x)
        
        if self.final:
            new_shape = tf.concat([shape[:-2],[-1]],0)
            out = tf.reshape(f,new_shape)

        else:
            new_shape = tf.concat([shape[:-1],[-1]],0)
            out = tf.reshape(x,new_shape)

        return out

def getSequential(dims=[32,32,1],name=None,activation=None,final=True):

    final_list = []

    for idx,n in enumerate(dims):
        if final and idx == len(dims)-1:
            final_list.append(Dense(1, activation=None,name=f"{name}-{idx}"))
        else:
            final_list.append(Dense(n, activation=activation,name=f"{name}-{idx}"))

    return tf.keras.Sequential(final_list, name=name)