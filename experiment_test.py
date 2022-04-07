import numpy as np
import gc
from data_sampling.load_gens import fixedTestSet
gc.enable()

def run_experiment(args, model, ds_names, mse):
    ts_bulk  = []

    tssf = []
    text = ""

    for fs in range(args.min_ft,args.max_ft+1):
        # ts_loss  = []
        # trues    = []
        test_gen = fixedTestSet(args,f_size=fs)
        tss = []
        for i in test_gen:
            ts_loss=model.predict(x=i[0])
            trues=i[1]
            tss.append(mse(ts_loss,trues))

        del(test_gen)
        gc.collect()

        tssf.append(np.array(tss))
        tss = np.array(tss)

        tss = np.reshape(tss,[-1,tss.shape[-2]])
        ts_mean     = np.mean(tss,axis=0)
        ts_std      = np.std( tss,axis=0)
        overall     = np.mean(ts_mean)
        overall_std = np.std(ts_mean)

        text = text+f"Features: {fs} (loss: {overall} +/- {overall_std}) \n"
        for i in range(len(ts_mean)):
            text = text+f"     {ts_mean[i]:.6f} +/- {ts_std[i]:.6f} {ds_names[-1][i].split('/')[-1]}\n"
        text = text+"\n"
        ts_bulk.append([fs,overall, overall_std, ts_mean, ts_std])

    tss_ag = np.array(tssf)
    print (tss_ag.shape)
    tss = np.reshape(tss_ag,[-1,tss_ag.shape[-2]])
    print (tss.shape)
    
    ts_mean     = np.mean(tss,axis=0)
    ts_std      = np.std( tss,axis=0)
    overall     = np.mean(ts_mean)
    overall_std = np.std(ts_mean)

    text2 = "Overall per DS: \n"
    for i in range(len(ts_mean)):
        text2 = text2+f"     {ts_mean[i]:.6f} +/- {ts_std[i]:.6f} {ds_names[-1][i].split('/')[-1]}\n"
        
    return [overall, overall_std, ts_mean, ts_std, tss_ag, ts_bulk, text2+"\n"+text]
