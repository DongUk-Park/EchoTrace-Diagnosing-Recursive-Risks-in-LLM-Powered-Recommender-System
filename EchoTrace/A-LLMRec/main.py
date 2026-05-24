import os
import sys
import argparse
from utils import *
from train_model import *

# from pre_train.sasrec.data_preprocess import preprocess

def allmrec(args, pos_item_id=0, phase=0, user_id = False, user_train = False, user_gt = False, model=None, seed=None):
    if phase != 0:
        args.phase = phase
    
    if args.phase == 1:
        train_model_phase1(args)
    elif args.phase == 2:
        train_model_phase2(args)
    elif args.phase == 3:
        #inference(0, 0, args)
        return user_inference(user_id, pos_item_id, user_train, user_gt, args, model, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='opt, llama')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='ml-100k')
    
    # train phase setting
    parser.add_argument('--phase', type=int, default=0)
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    args = parser.parse_args()
    print(f"device num : {args.gpu_num}")
    args.device = 'cuda:' + str(args.gpu_num)
    
    #args.inference = True
    #args.phase = phase
    args.rec_pre_trained_data = "ml-100k" #"Movies_and_TV"
    
    phases = [1, 2, 3]
    for phase in phases:
        allmrec(args, phase) 
    