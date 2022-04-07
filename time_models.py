import tensorflow as tf
from tensorflow.keras import Model



class SliceEncoderModel(Model):
    def __init__(self, control=80):
        
        super(SliceEncoderModel, self).__init__()
        self.control = control
        self.mul     = 1.0
        if control == 100:
            self.control = 1
            self.mul     = 0.0

    def call(self, x):
        #input is TASKS x SAMPLES x TIME X FEATURES
        
        t1 = x[:,:,-1,:-1]
        
        t2 = x[:,:,-self.control-1,-1:]*self.mul
        # t2 = tf.expand_dims(t2,-1)
        
        #reshaped to  TASKS X SAMPLES X FEATURES
        
        return tf.concat([t1,t2],axis=-1)