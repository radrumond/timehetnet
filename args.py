"""
Command-line argument parsing.
"""

import argparse
import time


# Used to parse boolean arguments
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
# args
def argument_parser():
    """
    Get an argument parser for a training script.
    """
     
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ####### Data args ##########
    parser.add_argument('--data_dir', help='Directory which contains data folders', default="~/data", type=str)
    parser.add_argument('--split_dir', help='Directory which contains data folders', default="./splits", type=str)
    
    
    parser.add_argument('--min_f', help='Minimum number of channels to sample', default=5   , type=int)
    parser.add_argument('--max_f', help='Maximum number of channels to sample', default=10   , type=int)

    parser.add_argument('--min_ft', help='Minimum number of channels to sample in test', default=5   , type=int)
    parser.add_argument('--max_ft', help='Maximum number of channels to sample in test', default=10   , type=int)
    parser.add_argument('--q_shots', help='Number of query samples per task', default=20   , type=int)

    parser.add_argument('--s_shots', help='Number of support samples per task', default=20   , type=int)
    parser.add_argument('--t_length', help='(Minimum) length of time series to sample; if no maximum length is given, it is fixed to this', default=100   , type=float)
    parser.add_argument('--tmax_length', help='Maximum length of time series to sample; if None, it is fixed to t_length', default=100   , type=float)
    parser.add_argument('--normalize_data' , help='Activates data normalization on the task batch generator', default="True", type=boolean_string)
    parser.add_argument('--better_norm' , help='Activates better data normalization on the task batch generator', default="True", type=boolean_string)
    
    parser.add_argument('--grad_clip' , help='Activates gradient clipping', default=0.0, type=float)
    parser.add_argument('--key', help='Keyname to group experiments', default="", type=str)
    
    
    
    ####### Model args #########

    parser.add_argument('--dims',        help='A list of model weights for hetnet', default="[32,32,32]", type=str)
    parser.add_argument('--dims_pred',   help='A list of model weights for hetnet pred', default="[32,32,32]", type=str)
    parser.add_argument('--hetmodel',    help='Type of hetnet (normal or time)', default="time", type=str)
    parser.add_argument('--batchnorm',  help='Batchnorm for TimeHetNet',  default="False" , type=boolean_string)
    parser.add_argument('--block',  help='Block type for TimeHetNet',  default="gru,conv,conv,gru" , type=str)
    
    
    ####### Probelm Type args #########

    parser.add_argument('--control_mode',  help='Define problems as control problems',  default="True" , type=boolean_string)
    parser.add_argument('--control_steps', help='Control steps for control problems', default=80   , type=int)

    ####### Training args ######
    parser.add_argument('--lr', help='Learning rate for optimizer', default=0.001   , type=float)
    parser.add_argument('--split', help='split index we wish to use', default=0   , type=int)
    parser.add_argument('--num_epochs', help='Number of training epochs', default=15000   , type=int)
    
    
    ####### Early Stopping######
    parser.add_argument('--early_stopping', help='decided if you want o early stop', default="True"   , type=boolean_string)
    parser.add_argument('--patience'   , help='patience for early stopping', default=1500   , type=int)
    parser.add_argument('--best_weights', help='Load best weights during early stopping' , default="True"   , type=boolean_string)
    
    
    
    ####### Results args #######
    parser.add_argument('--name', help='Experiment name for prints', default=None   , type=str)
    parser.add_argument('--res_dir', help='Directory to save results', default="./results/"   , type=str)
    parser.add_argument('--save_weights', help='decided if you want save the weights', default="True"   , type=boolean_string)


    return parser.parse_args()