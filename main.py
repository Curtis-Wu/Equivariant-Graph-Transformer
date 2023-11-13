import yaml
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from models.FinalModel import EGTF
from models.Trainer import Trainer

def create_model(model_dict):
    model = EGTF(
        # EGNN/EGCL parameters
        hidden_channels = model_dict["hidden_channels"],
        num_edge_feats = model_dict["num_edge_feats"],
        num_egcl = model_dict["num_egcl"],
        act_fn = model_dict["act_fn"],
        residual = model_dict["residual"],
        attention = model_dict["attention"],
        normalize = model_dict["normalize"],
        max_atom_type = model_dict["max_atom_type"],
        cutoff = model_dict["cutoff"],
        max_num_neighbors = model_dict["max_num_neighbors"],
        static_coord = model_dict["static_coord"],
        # Encoder Parameters
        d_model = model_dict["d_model"],
        num_encoder = model_dict["num_encoder"],
        num_heads = model_dict["num_heads"],
        num_ffn = model_dict["num_ffn"],
        act_fn_ecd = nn.SiLU(),
        dropout_r = model_dict["dropout_r"],
        # Energy head parameter
        num_neurons = model_dict["num_neuron"]
    )
    return model

if __name__ == "__main__":

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # Get correct activation function
    act_fn_dict = {"SiLU": nn.SiLU(), "ReLU": nn.ReLU}
    config["model_dict"]["act_fn"] = act_fn_dict[config["model_dict"]["act_fn"]]
    config["model_dict"]["act_fn_ecd"] = act_fn_dict[config["model_dict"]["act_fn_ecd"]]

    print(config)
    model = create_model(config["model_dict"])
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters is {total_params}")
    
    if config["load_model"]:
        # Load pre-trained model
        pretrained_model = torch.load(config["load_model"], map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_model, strict=False);
    
    trainer = Trainer(model, config)
    trainer.train()