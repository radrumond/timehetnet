from data_sampling.data_gen_ts    import Task_Sampler
import numpy as np

def getGens (args,model_type,load_test=False, onlyTest=False):
    split_idx = args.split
    print(f"Working on SPLIT: {split_idx}")
    splits = np.load(f"{args.split_dir}/splits.npy",allow_pickle=True)[split_idx]
    
    print("Loading data ...")
    data_loader = Task_Sampler(args.data_dir,splits)
    print("Data loaded ...")
    test_gen = None
    if onlyTest:
        train_gen = None
        val_gen   = None
    else:
        train_gen = data_loader.generateSet (args.min_f,
                                             args.max_f,
                                             args.s_shots,
                                             args.q_shots,
                                             augment=False,
                                             length=args.t_length,
                                             normalize=args.normalize_data,
                                             max_length=args.tmax_length,
                                             mode="train",
                                             better_norm=args.better_norm,
                                             control = args.control_mode,
                                             control_steps = args.control_steps
                                            )

        val_gen   = data_loader.generateSet (args.min_f,
                                             args.max_f,
                                             args.s_shots,
                                             args.q_shots,
                                             augment=False,
                                             length=args.t_length,
                                             normalize=args.normalize_data,
                                             max_length=args.tmax_length,
                                             mode="val",
                                             better_norm=args.better_norm,
                                             control=args.control_mode,
                                             control_steps=args.control_steps
                                             )
    if load_test:
        test_gen   = data_loader.generateSet (args.min_f,
                                         args.max_f,
                                         args.s_shots,
                                         args.q_shots,
                                         augment=False,
                                         length=args.t_length,
                                         normalize=args.normalize_data,
                                         max_length=args.tmax_length,
                                         mode="test",
                                         shuffle = False,
                                         better_norm=args.better_norm,
                                         control = args.control_mode,
                                         control_steps = args.control_steps
                                                )
            
    return train_gen, val_gen, test_gen, splits


            
def fixedTestSet(args,f_size=5): #,forecast=False):
    split = args.split
    # problemType = ""
    # if forecast:
    #     problemType = "_forecast"
    if args.control_mode:
        problemType = "_control"
        better = ""
        if args.better_norm:
            better = "_better"
        data_set = np.load(f"{args.split_dir}/{split}_control_test{better}/control_test{f_size}.npy", allow_pickle=True)
        print(f"Test loaded (control)! ({f_size}) ({better})")
        
    else:
        # data_set = np.load(f"{args.split_dir}/{split}_test{problemType}/test{f_size}{problemType}.npy", allow_pickle=True)
        print("please set control_mode to true, control_mode False is not supported anymore")
        exit(0) 
       
    for mb in data_set:
        if args.control_mode:
            ((qx,sx,sy),qy) = mb
            if args.control_steps > 0:
                sx[:,:,-args.control_steps:,-1] = 0
                qx[:,:,-args.control_steps:,-1] = 0
            # print("aqui",args.control_steps,sx[0])
            yield ((qx,sx,sy),qy)
        else:
            yield mb
    # del(data_set)

