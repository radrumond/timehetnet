
# TimeHetNet
### Written by: Rafael Rego Drumond and Lukas Brinkmeyer
### Co-written by: Johannes Burchert and Lars Schmidt-Thieme

If you use this code please cite
```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.03456,
  doi = {10.48550/ARXIV.2204.03456},
  url = {https://arxiv.org/abs/2204.03456},
  author = {Brinkmeyer, Lukas and Drumond, Rafael Rego and Burchert, Johannes and Schmidt-Thieme, Lars},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, 68},
  title = {Few-Shot Forecasting of Time-Series with Heterogeneous Channels},
  publisher = {arXiv},
  year = {2022}
}
```

This code also provide an implementation for the network of Tomoharu Iwata from the paper below (we nickname it HetNet).
```
@article{iwata2020meta,
  title={Meta-learning from tasks with heterogeneous attribute spaces},
  author={Iwata, Tomoharu and Kumagai, Atsutoshi},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6053--6063},
  year={2020}
}
```

####  Currently it only supports Linux

This is our implementation of TimeHetNet for our paper "Few-Shot Forecasting of Time-Series with Heterogeneous Channels".

Our conda environment has been exported to thn.yaml

In order to run our code you need to download the data-sets as listed in our paper.

Pre-processing code will be available for the camera ready version. The default directory is ``~/data``
For now, you can look into the file splits/regensplits.py on how the filenames must look like. ".npy" files are the data-sets converted to a numpy array (SAMPLESxTIMExCHANNELS). ".pkl.npy" are the same, but indicates data-sets where the size  of TIME is not the consistent across SAMPLES.

Once the data-set is processed, you must run:
```bash
python generate_test_set/generate_test_set.py
```
to generate fixed test sets. Keep in mind this process is random and will differ from our current experiments.

once this is done you can run:

```bash
python experiment.py
```
make sure you test the ``args.py`` file to match the same hyper parameters as used in the paper. You might wanna change the following ones:

```
--grad_clip     (0.0 to deactivate, 1.0 to use it as in the paper)
--dims          (the dimensions for our inference network. You can either set to '[16,16,16]' or '[32,32,32]')
--dims_pred     ( the dimensions for our prediction network. You should set it to '[32,32,32]')
--hetmodel      ('time' for our proposed TimeHetNet, 'Normal' for Iwata's hetnet)
--block         ('gru,conv,conv,gru' is our main architecture)
--control_steps (if you wish for example to run \(t_0 + 80\) experimets, set this to 80)
--split         (a number from 0 to 4, we have a 5-fold cross validation split. This is already defined in our original code)
```


