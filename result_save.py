import numpy as np
import os
import datetime


def save_results(args, history, ts_loss=None, key=None, randomnumber=None):
    
    sha  = None
    # if args.isGit:
    #     import git
    #     repo = git.Repo(search_parent_directories=True)
    #     sha  = repo.head.object.hexsha
    
    to_save = {
                    "args"         : args,
                    "loss"         : history.history['loss'],
                    "loss_std"     : history.history['metricSTD'],
                    "val_loss"     : history.history['val_loss'],
                    "val_std"      : history.history['val_metricSTD'],
                    "test_loss"    : ts_loss,
                    "git_hash"     : sha
              }
    if randomnumber is None:
        randomnumber  = int(np.random.rand(1)*10000000)
    date = datetime.datetime.now().strftime("%b-%d-%Y-%H:%M:%S")
    if key is not None:
        date = key
    os.system(f"mkdir -p ./results/")
    os.system(f"mkdir -p ./results/{date}")
    
    np.save(f"./results/{date}/{args.split}_{randomnumber}_results.npy",to_save)
    
    conf = ""
    for arg in vars(args):
        conf = conf+ f"{arg} : {str(getattr(args, arg))}\n"
    conf = conf + f"git Hash: {sha}\n"
    conf = conf + f"Test  loss final :{ts_loss[0]} +/- {ts_loss[1]}\n"
    conf = conf + f"Train loss final :{history.history['loss'][-1]} +/- {history.history['metricSTD'][-1]} \n"
    conf = conf + f"Val   loss final :{history.history['val_loss'][-1]} +/- {history.history['val_metricSTD'][-1]}\n"
    conf = conf + "\n" + ts_loss[-1]
    for b in ts_loss[-2]:
        conf = conf + f"\n {b}"
    with open(f"./results/{date}/{args.split}_{randomnumber}_args.txt", "w") as text_file:
        text_file.write(conf)
    return f"./results/{date}/{randomnumber}"
