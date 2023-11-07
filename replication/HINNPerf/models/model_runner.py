from utils.data_preproc import system_samplesize, seed_generator, DataPreproc
import numpy as np
import time
import tensorflow as tf

class ModelRunner():
    """Generic class for training models"""

    def __init__(self, data_preproc, model_class):
        """
        Args:
            data_preproc: [DataPreproc object] preprocess and generate training data
            model_class: [class] deep learning model class
        """
        self.data_preproc = data_preproc
        self.model_class = model_class
    
    def train(self, config, sample_size, number_experiment, cur_exp):
        """
        Train the model

        Args:
            config: configures to create a model object
        """
        N_train = sample_size
        #print("Sample size: {}".format(N_train))

        if (N_train >= self.data_preproc.all_sample_num):
            raise AssertionError("Sample size can't be larger than whole data")

        # Get the initial seed
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)

        # Set seed and generate training data
        seed = seed_init * number_experiment + cur_exp
        if (config['gnorm']):
            X_train, Y_train, X_valid, Y_valid, mean_Y, std_Y = self.data_preproc.get_train_valid_samples(N_train, seed, config['gnorm'])
        else:
            X_train, Y_train, X_valid, Y_valid, max_Y = self.data_preproc.get_train_valid_samples(N_train, seed, config['gnorm'])

        model = self.model_class(config)
        #print("Start training " + model.name + "...")
        model.build_train()
        
        lr = config['lr']
        decay = lr/1000
        train_seed = 0
        for epoch in range(1, 2000):
            train_seed += 1
            _, cur_loss, pred = model.sess.run([model.train_op, model.loss, model.output],
                                               {model.X:X_train, model.Y:Y_train, model.lr:lr})
                        
            #if epoch % 500 == 0 or epoch == 1:
            #    rel_error = np.mean(np.abs(np.divide(Y_train.ravel() - pred.ravel(), Y_train.ravel())))
            #    if model.verbose:
            #        print("Cost function: {:.4f}", cur_loss)
            #        print("Train relative error: {:.4f}", rel_error)
                        
            lr = lr*1/(1 + decay*epoch)
                
        Y_pred_train = model.sess.run(model.output, {model.X: X_train})
        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train))

        Y_pred_val = model.sess.run(model.output, {model.X: X_valid})
        if (config['gnorm']):
            Y_pred_val = Y_pred_val * (std_Y + (std_Y == 0) * .001) + mean_Y
            Y_valid = Y_valid * (std_Y + (std_Y == 0) * .001) + mean_Y
        else:
            Y_pred_val = Y_pred_val * max_Y
            Y_valid = Y_valid * max_Y
        abs_error_val = np.mean(np.abs(Y_pred_val - Y_valid))

        model.finalize()

        return abs_error_train, abs_error_val

    def test(self, best_config, sample_size, number_experiment, cur_exp):
        N_train = sample_size

        # Get the initial seed
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)

        # Set seed and generate training data
        seed = seed_init * number_experiment + cur_exp
        if best_config['gnorm']:
            X_train, Y_train, X_test, Y_test, mean_Y, std_Y = self.data_preproc.get_train_test_samples(N_train, seed, best_config['gnorm'])
        else:
            X_train, Y_train, X_test, Y_test, max_Y = self.data_preproc.get_train_test_samples(N_train, seed, best_config['gnorm'])

        model = self.model_class(best_config)
        #print("Retraining " + model.name + " and testing ...")
        model.build_train()

        lr = best_config['lr']
        decay = lr/1000
        train_seed = 0
        for epoch in range(1, 2000):
            train_seed += 1
            _, cur_loss, pred = model.sess.run([model.train_op, model.loss, model.output],
                                               {model.X:X_train, model.Y:Y_train, model.lr:lr})
            
            if epoch % 500 == 0 or epoch == 1:
                rel_error = np.mean(np.abs(np.divide(Y_train.ravel() - pred.ravel(), Y_train.ravel())))
                if model.verbose:
                    print("Cost function: {:.4f}", cur_loss)
                    print("Train relative error: {:.4f}", rel_error)
            
            lr = lr*1/(1 + decay*epoch)
        
        Y_pred_test = model.sess.run([model.output], {model.X: X_test})

        if best_config['gnorm']:
            Y_pred_test = Y_pred_test * (std_Y + (std_Y == 0) * .001) + mean_Y
        else: 
            Y_pred_test = max_Y * Y_pred_test
        rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel())))
        print('Prediction relative error (%): {:.2f}'.format(np.mean(rel_error)*100))

        model.finalize()

        return np.mean(rel_error)*100
