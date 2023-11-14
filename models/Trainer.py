import os
import gc
import csv
import time
import torch
import shutil
import numpy as np

from datetime import datetime
from torch.optim import AdamW
import torch.multiprocessing
import torch.nn.functional as F
from models.DataLoaders import ANI1Wrapper
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Normalizer class for normalizing and denormalizing energy values
class Normalizer(object):
    """Class for normalization and de-normalization of tensors. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

# Trainer class
class Trainer(object):

    def __init__(self, model, config):
        # Get config and device
        self.model = model
        self.config = config
        self.device = self._get_device()
        # Data processing using ANIWrapper
        self.dataset = ANI1Wrapper(**self.config['dataset_dict'])
        # Prefix for log directory names
        self.prefix = 'ani1'
        self.model_prefix = 'egtf'
        dir_name = '_'.join([datetime.now().strftime('%b%d_%H-%M-%S'), self.prefix, self.model_prefix])
        self.log_dir = os.path.join('Runs', dir_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    # Get device function
    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    # config saving
    @staticmethod
    def _save_config_file(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            shutil.copy('./config.yaml', os.path.join(ckpt_dir, 'config.yaml'))

    # loss function
    def loss_fn(self, model, data):
        data = data.to(self.device)
        pred_e = model(data.x, data.pos, data.batch)
        loss = F.mse_loss(
            pred_e, self.normalizer.norm(data.y), reduction='mean'
        )
        return pred_e, loss

    def train(self):
        # load training, validation, test loader
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        energy_labels = []
        for data in train_loader:
            energy_labels.append(data.y)
        labels = torch.cat(energy_labels)
        # normalize energy values
        self.normalizer = Normalizer(labels)
        gc.collect() # free memory

        model = self.model.to(self.device)

        # Read in training learning rate parameters
        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        
        optimizer = AdamW(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(ckpt_dir)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        from models.utils import adjust_learning_rate  # This is a function that adjusts learning rate according to
                                                # specified lr, min_lr, epochs, warmup_epochs, patience_epochs

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                # adjust learning rate accordingly                
                adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)
                # use custom loss function because of normalization
                __, loss = self.loss_fn(model, data)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record training loss and current learning rate
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    print(f"Training at Epoch: {epoch_counter+1}, Batch #: {bn+1}, RMSE: {loss.item()}")
                    torch.cuda.empty_cache()

                if n_iter != 0 and n_iter % 1000 == 0:
                    valid_rmse = self._validate(model, valid_loader)
                    self.writer.add_scalar('valid_rmse', valid_rmse, global_step=n_iter)
                    print(f"Validation at Epoch: {epoch_counter+1}, RMSE: {valid_rmse}")

                    if valid_rmse < best_valid_loss:
                        best_valid_loss = valid_rmse
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                n_iter += 1
            
            gc.collect() # free memory
            torch.cuda.empty_cache()

            # validate the model 
            # valid_rmse = self._validate(model, valid_loader)
            # self.writer.add_scalar('valid_rmse', valid_rmse, global_step=valid_n_iter)
            # print(f"Validation at Epoch: {epoch_counter+1}, RMSE: {valid_rmse}")

            # if valid_rmse < best_valid_loss:
            #     best_valid_loss = valid_rmse
            #     torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))

            valid_n_iter += 1
        
        start_time = time.time()
        self._test(model, test_loader)
        print('Test duration:', time.time() - start_time)

    # Validation set function
    def _validate(self, model, valid_loader):
        predictions, labels = [], []
        model.eval()

        for bn, data in enumerate(valid_loader):
            pred_e, loss = self.loss_fn(model, data)
            pred_e = self.normalizer.denorm(pred_e)

            y = data.y

            if self.device == 'cpu':
                predictions.extend(pred_e.flatten().detach().numpy())
                labels.extend(y.flatten().numpy())
            else:
                predictions.extend(pred_e.flatten().cpu().detach().numpy())
                labels.extend(y.cpu().flatten().numpy())
            
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        model.train()
        return mean_squared_error(labels, predictions, squared=False)
    
    # Test set function
    def _test(self, model, test_loader):
        # Load the best validation model
        model_path = os.path.join(self.log_dir, 'checkpoints', 'best_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print(f"Loaded {model_path} successfully.")
        
        predictions, labels = [], []
        model.eval()

        for bn, data in enumerate(test_loader):                
            pred_e, _ = self.loss_fn(model, data)
            pred_e = self.normalizer.denorm(pred_e)

            label = data.y

            # add the self interaction energy back
            pred_e += data.self_energy
            label += data.self_energy

            if self.device == 'cpu':
                predictions.extend(pred_e.flatten().detach().numpy())
                labels.extend(label.flatten().numpy())
            else:
                predictions.extend(pred_e.flatten().cpu().detach().numpy())
                labels.extend(label.cpu().flatten().numpy())
        
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        rmse = mean_squared_error(labels, predictions, squared=False)
        mae = mean_absolute_error(labels, predictions)
        print(f"The test RMSE and MAE are {rmse}, {mae}")

        # Write the 
        with open(os.path.join(self.log_dir, 'results.csv'), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(labels)):
                csv_writer.writerow([predictions[i], labels[i]])
            csv_writer.writerow([rmse, mae])