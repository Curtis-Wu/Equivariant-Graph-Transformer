import gc
import yaml
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from main import create_model
from models.Trainer import Normalizer
from models.DataLoaders import ANI1Wrapper
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluater(object):

    def __init__(self, log_dir):
        
        self.log_dir = log_dir
        self.ckpt_dir = log_dir + "/checkpoints"
        self.config = yaml.load(open(self.ckpt_dir + "/config.yaml", "r"), Loader=yaml.FullLoader)
        self.config["dataset_dict"]["data_dir"] = "./Data_eval"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = ANI1Wrapper(**self.config['dataset_dict'])
        # Create model
        self._create_model()
        self.scale_value = self.config.get("scale_value", 1)

        self.normalize_energies = self.config["normalize_energies"]
        if self.normalize_energies:
            self.load_normalizer_values()
    
    def _create_model(self):
        # Create and load pre-trained model
        act_fn_dict = {"SiLU": nn.SiLU(), "ReLU": nn.ReLU}

        self.config["model_dict"]["act_fn"] = act_fn_dict[self.config["model_dict"]["act_fn"]]
        self.config["model_dict"]["act_fn_ecd"] = act_fn_dict[self.config["model_dict"]["act_fn_ecd"]]

        self.model = create_model(self.config["model_dict"])
        self._load_model()

    def _load_model(self):
        # Load pre-trained model
        model_path = self.ckpt_dir + "/best_model.pth"

        pretrained_model = torch.load(model_path, map_location = self.device)
        self.model.load_state_dict(pretrained_model, strict=False)

        print(f"Successfully loaded pre-trained model {model_path}")

    def loss_fn(self, model, data):
        data = data.to(self.device)
        pred_e = model(data.x, data.pos, data.batch)

        if self.normalize_energies:
            loss = F.mse_loss(
                pred_e, self.normalizer.norm(data.y), reduction='mean')
        else:
            loss = F.mse_loss(
                pred_e, data.y/self.scale_value, reduction='mean')

        return pred_e, loss
    
    def load_normalizer_values(self):
        # Load mean and std for training data
        dummy_tensor = torch.zeros(1)
        self.normalizer = Normalizer(dummy_tensor)

        with open(self.log_dir + "/normalizer_values.json", 'r') as f:
            data = json.load(f)
            self.normalizer.load_state_dict(data)

    def evaluate(self):
        
        _, _, test_loader = self.dataset.get_data_loaders()

        model = self.model.to(self.device)
        predictions, labels = [], []
        model.eval()

        for bn, data in enumerate(test_loader):   
            pred_e = model(data.x, data.pos, data.batch)

            if self.normalize_energies:
                pred_e = self.normalizer.denorm(pred_e)
            else: pred_e *= self.scale_value

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
        print(f"The evaluated RMSE and MAE are {rmse}, {mae}")    
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Pass in log directory as argument to evaluate.")
    parser.add_argument("log_dir", help = "The log directory.", type = str)
    args = parser.parse_args()
    log_dir = args.log_dir
    
    evaluater = Evaluater(log_dir)
    evaluater.evaluate()