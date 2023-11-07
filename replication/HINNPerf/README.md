# HINNPerf: Hierarchical Interaction Neural Network for Performance Prediction of Configurable Systems

This drive releases the code and data for the HINNPerf model.

## Dependencies

+ Python 3.7.6
+ Numpy 1.20.3
+ Tensorflow 1.15.0
+ Seaborn 0.11.2
+ Matplotlib 3.5.1

## Directories

+ `datasets`: dataset including the performance data of the ten subject systems.
+ `models`: python code for building, training and testing the HINNPerf model.
+ `utils`: python code for auxiliary training.
+ `results`: the prediction results of HINNPerf on the ten subject systems.

## Usage

To run HINNPerf, users need to specify the name of the software system they wish to evaluate and then run the script `run.py`. There are 10 software systems that users can evaluate: x264, BDBJ, lrzip, vp9, polly, Dune, hipacc, hsmgp, javagc, sac. The script will then evaluate HINNPerf on the chosen software system with the same experiment setup presented in our paper. Specifically, for binary software systems, DeepPerf will run with five different sample sizes: n, 2n, 4n, 6n with n being the number of options, and 30 experiments for each sample size. For binary-numeric software systems, HINNPerf will run with the sample sizes specified in Table 3 of our paper, and 30 experiments for each sample size. For example, if users want to evaluate HINNPerf with the system x264, the command line to run HINNPerf will be:

```shell
$ python run.py x264
```

When finishing each sample size, the script will output a .csv file that shows the mean prediction error and the margin (95% confidence interval) of that sample size over the 30 experiments. These results will be similar as the results we report in Table 3 of our paper.

Alternatively, users can customize the sample size and/or the number of experiments for each sample size by using the optional arguments `-ss` and `-ne`. For example, to set the sample size = 20 and the number of experiments = 10, the corresponding command line is:

```shell
$ python run.py x264 -ss 20 -ne 10
```

Setting none or one option will result in the other option(s) running with the default setting. The default setting of the number of experiments is 30. The default setting of the sample size is: (a) the four different sample sizes: n, 2n, 4n, 6n, with n being the number of configuration options, when the evaluated system is a binary system OR (b) the four sample sizes specified in Table 3 of our paper when the evaluated system is a binary-numeric system.
