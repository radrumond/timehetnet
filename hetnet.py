import os


import numpy as np
import tensorflow as tf


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.activations import softmax, relu


import os
import numpy as np
import time
tf.keras.backend.set_floatx('float32')





def getSequential(dims=[32,32,1],name=None,activation=None,final=True):

    final_list = []

    for idx,n in enumerate(dims):
        if final and idx == len(dims)-1:
            final_list.append(Dense(1, activation=None,name=f"{name}-{idx}"))
        else:
            final_list.append(Dense(n, activation=activation,name=f"{name}-{idx}"))

    return tf.keras.Sequential(final_list, name=name)


class HetNet(tf.keras.Model):

    def __init__(self, enc,
                       enc_type="slice",
                       dims = [32,32,32],
                       acti="relu",
                       drop1=0.1,
                       drop2=0.1,
                       share_qs=False,
                       base = True):
        
        super(HetNet, self).__init__()
        self.enc      = enc
        self.enc_type = enc_type
        self.base = base
        # Prediction network
        self.dense_fy = getSequential(dims=dims,activation=acti,final=True,name="pred_fy")
        self.dense_fz = getSequential(dims=dims,activation=acti,final=False,name="pred_fz")
        self.dense_gz = getSequential(dims=dims,activation=acti,final=False,name="pred_gz")

        # Support and Query network (start with both same weights)
        self.dense_fv = getSequential(dims=dims,activation=acti,final=False,name="s_dense_fv")
        self.dense_gv = getSequential(dims=dims,activation=acti,final=False,name="s_dense_gv")

        # U net
        self.dense_uf = getSequential(dims=dims,activation=acti,final=False,name="ux_dense_f")
        self.dense_ug = getSequential(dims=dims,activation=acti,final=False,name="ux_dense_g")

        # Vbar network
        self.dense_v  = getSequential(dims=dims,activation=acti,final=False,name="vb_dense_v")
        self.dense_c  = getSequential(dims=dims,activation=acti,final=False,name="vb_dense_c")

        self.drop_layer1=tf.keras.layers.Dropout(drop1)
        self.drop_layer2=tf.keras.layers.Dropout(drop2)
        self.drop_layer3=tf.keras.layers.Dropout(drop2)


    def sub_call2(self, inp, training=False):
        #que_x, sup_x, sup_y, dummy_i, dummy_n = inp
        que_x, sup_x, sup_y = inp
        
        
        ##### Vbar network #####
        # Encode sup_x to FxK
        vs_bar = tf.expand_dims(sup_x,axis=-1)      
        vs_bar = self.dense_v(vs_bar)
        vs_bar = tf.reduce_mean(vs_bar, axis=1)
        vs_bar = self.dense_c(vs_bar)
        # print(vs_bar.shape)

        if not self.base:
            # Encode que_x to FxK
            vq_bar = tf.expand_dims(que_x,axis=-1)      
            vq_bar = self.dense_v(vq_bar)
            vq_bar = tf.reduce_mean(vq_bar, axis=1)
            vq_bar = self.dense_c(vq_bar)
            # print(vq_bar.shape)

        # Encode sup_y to FxK
        cs_bar = tf.expand_dims(sup_y,axis=-1)      
        cs_bar = self.dense_v(cs_bar)
        cs_bar = tf.reduce_mean(cs_bar, axis=1)
        cs_bar = self.dense_c(cs_bar)
        # print(cs_bar.shape)
        
        ##### U network ##### 
        # Tile FxK to NxFxK or NxJxK respectively
        vs_bar = tf.tile(tf.expand_dims(vs_bar,axis=1),[1,tf.shape(sup_x)[1],1,1])
        if not self.base:
            vq_bar = tf.tile(tf.expand_dims(vq_bar,axis=1),[1,tf.shape(que_x)[1],1,1])
        cs_bar = tf.tile(tf.expand_dims(cs_bar,axis=1),[1,tf.shape(sup_y)[1],1,1])
        # print(vs_bar.shape,vq_bar.shape,cs_bar.shape)
        
        # Concatenate tiled to NxFxK+1 or NxJxK+1 respectively
        u_xs = tf.concat([tf.cast(tf.expand_dims(sup_x,axis=-1),dtype=tf.float32),vs_bar],axis=-1)
        if not self.base:
            u_xq = tf.concat([tf.cast(tf.expand_dims(que_x,axis=-1),dtype=tf.float32),vq_bar],axis=-1)
        u_ys = tf.concat([tf.cast(tf.expand_dims(sup_y,axis=-1),dtype=tf.float32),cs_bar],axis=-1)
        # print(u_xs.shape,u_xq.shape,u_ys.shape)
        
        # Embed latent
        u_xs = self.dense_uf(u_xs)
        if not self.base:
            u_xq = self.dense_uf(u_xq)
        u_ys = self.dense_uf(u_ys)
        u_xs = tf.reduce_mean(u_xs, axis=2)
        if not self.base:
            u_xq = tf.reduce_mean(u_xq, axis=2)
        u_ys = tf.reduce_mean(u_ys, axis=2)
        # print(u_xs.shape,u_xq.shape,u_ys.shape)

        u_s  = u_xs + u_ys
        u_s  = self.dense_ug(u_s)
        if not self.base:
            u_q  = self.dense_ug(u_xq)
        # print(u_s.shape,u_q.shape)
        
        ##### Support network #####
        # Tile u features from NxK to NxFxK / NxJxK
        u_xs = tf.tile(tf.expand_dims(u_s,axis=2),[1,1,tf.shape(sup_x)[2],1])
        if not self.base:
            u_xq = tf.tile(tf.expand_dims(u_q,axis=2),[1,1,tf.shape(que_x)[2],1])
        u_ys = tf.tile(tf.expand_dims(u_s,axis=2),[1,1,tf.shape(sup_y)[2],1])
        # print("aaaaaah", u_xs.shape,u_xq.shape,u_ys.shape)
        
        # Concatenate with original and embed to NxFxK
        in_xs = tf.concat([tf.cast(tf.expand_dims(sup_x,axis=-1),dtype=tf.float32),u_xs],axis=-1)
        if not self.base:
            in_xq = tf.concat([tf.cast(tf.expand_dims(que_x,axis=-1),dtype=tf.float32),u_xq],axis=-1)
        in_ys = tf.concat([tf.cast(tf.expand_dims(sup_y,axis=-1),dtype=tf.float32),u_ys],axis=-1)
        in_xs = self.dense_fv(in_xs)
        if not self.base:
            in_xq = self.dense_fv(in_xq)
        in_ys = self.dense_fv(in_ys)

        # Aggregate and embed to FxK / JxK
        in_xs = tf.reduce_mean(in_xs, axis=1)
        if not self.base:
            in_xq = tf.reduce_mean(in_xq, axis=1)
        in_ys = tf.reduce_mean(in_ys, axis=1)
        in_xs = self.dense_gv(in_xs)
        if not self.base:
            in_xq = self.dense_gv(in_xq)
        in_ys = self.dense_gv(in_ys)

        # Dropout
        in_xs = self.drop_layer1(in_xs,training=training)
        if not self.base:
            in_xq = self.drop_layer2(in_xq,training=training)
        in_ys = self.drop_layer3(in_ys,training=training)

        ##### Prediction net ##### 
        # Tile support_net outputs to NxFxK / NxJxK
        #if True:
        p_xs = tf.tile(tf.expand_dims(in_xs, axis=1),[1,tf.shape(que_x)[1],1,1])
        if not self.base:
            p_xq = tf.tile(tf.expand_dims(in_xq, axis=1),[1,tf.shape(que_x)[1],1,1])
        p_ys = tf.tile(tf.expand_dims(in_ys, axis=1),[1,tf.shape(que_x)[1],1,1]) 

        if not self.base:
            z = tf.concat([tf.cast(tf.expand_dims(que_x, axis=-1), dtype=tf.float32),p_xs,p_xq],axis=-1)
        else:
            z = tf.concat([tf.cast(tf.expand_dims(que_x, axis=-1), dtype=tf.float32),p_xs],axis=-1)

        z = self.dense_fz(z) 
        z = tf.reduce_mean(z, axis=2) # Nq x K
        z = self.dense_gz(z)

        z = tf.tile(tf.expand_dims(z, axis=2),[1,1,tf.shape(sup_y)[2],1])
        y = tf.concat([z,p_ys],axis=-1)
        y = self.dense_fy(y)
        # return y
        return tf.squeeze(y, axis=-1)
    
    # input should be [samples X features] and [samples X labels]
    def call(self, inp, multi_task=True, training=False):

        que_x, sup_x, sup_y = inp
        

        
        que_x = self.enc(que_x)
        sup_x = self.enc(sup_x)
        
        inp = (que_x, sup_x, sup_y)

        if multi_task:
            return self.sub_call2(inp)
        return self.sub_call((que_x[0], sup_x[0], sup_y[0], dummy_i[0], dummy_n[0]))
        # return tf.map_fn(self.sub_call,elems=(que_x, sup_x, sup_y, dummy_i, dummy_n))