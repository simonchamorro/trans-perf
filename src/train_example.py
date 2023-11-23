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
    args = parser.parse_args()

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
                epochs = 1000,
            )
    
    model = Transperf(input_size=src_shape)
    seq_len = src_shape if src_shape % model.nhead == 0 else (src_shape // model.nhead)*model.nhead + model.nhead
    runner = ModelRunner(data_gen, model, seq_len=seq_len)
    mean_error, rel_error = runner.test(config, save_model=True)
    
    print('Mean prediction relative error (%) is: {:.2f}'.format(rel_error))