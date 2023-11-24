import torch
import numpy as np
from dataset import PerfDataset
from torch.utils.data import DataLoader
from data_preprocess import system_samplesize, seed_generator, DataPreproc
from tqdm import tqdm

from captum.attr import IntegratedGradients


class InterpretableModelRunner():
    def __init__(self, data_gen, model, batch_size=32):
        self.data_gen = data_gen
        self.model = model
        self.batch_size = batch_size
        
        # Check if GPU is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the specified device
        
    def train(self, config, sample_size=None, number_experiment=1, cur_exp=0):
        
        # Number of samples
        if sample_size is None:
            train_num = int(self.data_gen.Y_all.shape[0]*0.7)
        else:
            train_num = sample_size
        
        # Set seed
        seed_init = seed_generator(self.data_gen.sys_name, train_num)
        seed = seed_init * number_experiment + cur_exp
        
        if (config['gnorm']):
            x_train, y_train, x_valid, y_valid, _, _ = self.data_gen.get_train_valid_samples(train_num, seed, config['gnorm'])
        else:
            x_train, y_train, x_valid, y_valid, _ = self.data_gen.get_train_valid_samples(train_num, seed, config['gnorm'])
        
        train_dataset = PerfDataset(x_train, y_train, self.model.d_model)
        valid_dataset = PerfDataset(x_valid, y_valid, self.model.d_model)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model for 100 epochs
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=config['lr'],
                                     weight_decay=config['lr']/1000)
        for epoch in range(config['epochs']):
            self.model.train()
            error = 0.0
            num_samples = 0
            for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                output = self.model(x_train)
                loss = torch.nn.functional.mse_loss(output, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_error = torch.abs(output - y_train)
                error += batch_error.sum().item()
                num_samples += y_train.size(0)
            error_train = error / num_samples
            
            # Validation
            self.model.eval()
            error = 0.0
            rel_error = 0.0
            num_samples = 0
            for batch_idx, (x_valid, y_valid) in enumerate(valid_dataloader):
                x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
                output = self.model(x_valid)
                batch_error = np.abs(y_valid.detach().cpu().numpy().ravel() - output.detach().cpu().numpy().ravel())
                error += np.sum(batch_error)
                rel_error += np.sum(np.divide(batch_error, np.abs(y_valid.detach().cpu().numpy().ravel())))
                num_samples += y_valid.size(0)
            error_valid = error / num_samples
            rel_error_valid = rel_error / num_samples
        print("Valid - Epoch: {}, Mean error: {}, Mean rel error: {}".format(epoch, error_valid, rel_error_valid*100))
        return error_train, error_valid
    
    def test(self, config, sample_size=None, number_experiment=1, cur_exp=0, 
             save_model=False, train_model=True):
        
        # Number of samples
        if sample_size is None:
            train_num = int(self.data_gen.Y_all.shape[0]*0.7)
        else:
            train_num = sample_size
            
        # Model path
        save_path = "models/{}/model.pt".format(self.data_gen.sys_name)
        
        # Set seed
        seed_init = seed_generator(self.data_gen.sys_name, train_num)
        seed = seed_init * number_experiment + cur_exp
        
        y_mean = None
        y_std = None
        y_max = None
        if (config['gnorm']):
            x_train, y_train, x_test, y_test, y_mean, y_std = self.data_gen.get_train_test_samples(train_num, seed, config['gnorm'])
        else:
            x_train, y_train, x_test, y_test, y_max = self.data_gen.get_train_test_samples(train_num, seed, config['gnorm'])
        
        train_dataset = PerfDataset(x_train, y_train, self.model.d_model)
        test_dataset = PerfDataset(x_test, y_test, self.model.d_model)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model for n epochs
        num_epochs = config['epochs'] if train_model else 1
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=config['lr'],
                                     weight_decay=config['lr']/1000)
        for epoch in tqdm(range(num_epochs)):
            if train_model:
                self.model.train()
                error = 0.0
                num_samples = 0
                for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                    x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                    output = self.model(x_train)
                    loss = torch.nn.functional.mse_loss(output, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_error = torch.abs(output - y_train)
                    error += batch_error.sum().item()
                    num_samples += y_train.size(0)
                error_train = error / num_samples
            
            # test
            self.model.eval()
            error = 0.0
            rel_error = 0.0
            num_samples = 0
            for batch_idx, (x_test, y_test) in enumerate(test_dataloader):
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                output = self.model(x_test)
                output = self.post_process(output, config['gnorm'], y_max, y_mean, y_std)
                batch_error = np.abs(y_test.detach().cpu().numpy().ravel() - output)
                error += np.sum(batch_error)
                rel_error += np.sum(np.divide(batch_error, np.abs(y_test.detach().cpu().numpy().ravel())))
                num_samples += y_test.size(0)
            error_test = error / num_samples
            rel_error_test = rel_error / num_samples
        if save_model:
            self.model.save_model(save_path)
        print("Test - Epoch: {}, Mean error: {}, Mean rel error: {}".format(epoch, error_test, rel_error_test*100))
        
        ig = IntegratedGradients(self.model)
        input = x_test
        baseline = torch.zeros_like(input)
        attributions, convergence_delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
        print('IG Attributions:', attributions)
        print('Convergence Delta:', convergence_delta)

        return error_test, rel_error_test*100, attributions, convergence_delta
    
    def post_process(self, output, gnorm=False, max_y=None, mean_y=None, std_y=None):
        output = output.detach().cpu().numpy().ravel()
        if gnorm:
            output = output * (std_y + (std_y == 0) * .001) + mean_y
        else:
            output = max_y * output
        return output
