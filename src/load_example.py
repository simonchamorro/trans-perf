import argparse

from data_preprocess import DataPreproc
from model import Transperf
from model_runner import ModelRunner


if __name__ == '__main__':

    # Get system name from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("system_name",
                        help="name of system to be evaluated: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac",
                        type=str)
    parser.add_argument("-lm", "--load_model",
                        help="name of model to be loaded",
                        type=str,
                        default='test')
    args = parser.parse_args()
    
    # Model path
    LOAD_MODEL = "models/{}/{}.pt".format(args.system_name, args.load_model)

    # System to be evaluated:
    sys_name = args.system_name
    print(sys_name)
    data_gen = DataPreproc(sys_name)
    src_sample = data_gen.get_train_valid_samples(1, 1)
    src_shape = src_sample[0].shape[1]
    
    # Config
    config = dict(
                input_dim = data_gen.config_num,
                gnorm = False,
                lr = 0.001,
                epochs = 100,
            )
    
    nhead = 8
    model = Transperf(input_size=src_shape, nhead=nhead, load_model=LOAD_MODEL)
    if src_shape % nhead != 0:
        d_model = (src_shape // nhead)*nhead + nhead
    else:
        d_model = src_shape
    
    runner = ModelRunner(data_gen, model)
    mean_error, rel_error = runner.test(config, train_model=False)
    
    print('Mean prediction relative error (%) is: {:.2f}'.format(rel_error))