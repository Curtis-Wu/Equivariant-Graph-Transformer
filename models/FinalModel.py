import torch.nn as nn
from copy import deepcopy
from models.EGCL import E_GCL
from torch_scatter import scatter
from torch_cluster import radius_graph
from models.Transformer_Encoder import EncoderLayer

class EGTF(nn.Module):
    def __init__(self, # EGNN/EGCL parameters
                 hidden_channels, num_edge_feats, num_egcl, 
                 act_fn = nn.SiLU(), residual = True, attention = True,
                 normalize = False, max_atom_type = 100, cutoff = 5.0,
                 max_num_neighbors = 32, static_coord = True, freeze_egcl = True,
                 # Transformer-Encoder parameters
                 d_model = 256, num_encoder = 1, num_heads = 8,
                 num_ffn = 256, act_fn_ecd = nn.SiLU(), dropout_r = 0.1,
                 # Energy Head parameter
                 num_neurons = 512):

        super(EGTF, self).__init__()
        # self.hidden_channels = hidden_channels
        self.n_layers = num_egcl
        # self.max_atom_type = max_atom_type
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        # Create embeddings of dimension (hidden_channels, ) for each atom type
        self.type_embedding = nn.Embedding(max_atom_type, hidden_channels)
        self.freeze_egcl = freeze_egcl
        self.freeze_state = freeze_egcl
        
        # EGC layers
        for i in range(0, num_egcl):
            self.add_module("gcl_%d" % i, E_GCL(
                input_nf = hidden_channels, 
                output_nf = hidden_channels, 
                hidden_nf = hidden_channels, 
                add_edge_feats = num_edge_feats,
                act_fn = act_fn, residual = residual, 
                attention = attention, normalize = normalize,
                static_coord = static_coord))
            
        # Whether or not to freeze parameters of pre-trained egnn
        if freeze_egcl:
            for i in range(num_egcl-1):
                layer_name = f"gcl_{i}"
                for param in getattr(self, layer_name).parameters():
                    param.requires_grad = False
                    
        # Transformer-Encoder layers
        self.encoder_layers = \
            nn.ModuleList([EncoderLayer(d_model, num_heads,
                                        num_ffn, dropout_r, act_fn_ecd) 
                                        for _ in range(num_encoder)])

        # Energy Head
        self.energy_fc = nn.Sequential(
            nn.Linear(d_model, num_neurons),
            act_fn_ecd,
            nn.Linear(num_neurons, 1)
        )

    def forward(self, z, pos, batch, edge_index=None, edge_feats=None):
        h = self.type_embedding(z)
        x = deepcopy(pos)

        if edge_index is None:
            # Calculates edge_index from graph structure based on cutoff radius
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors + 1,
            )
        # EGC layers
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, x, add_edge_feats=edge_feats)
        # Encoder layers
        for layer in self.encoder_layers:
            h = layer(h)
        # Energy Head
        h = h.squeeze(0)  # Assuming the batch dimension is at dim 0
        h = scatter(h, batch, dim=0, reduce='add')
        
        out = self.energy_fc(h)

        return out
    
    def freeze_layers(self):
        if self.freeze_egcl:
            return
        for i in range(self.n_layers):
            layer_name = f"gcl_{i}"
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = False
        self.freeze_state = True
                    
    def unfreeze_layers(self):
        if self.freeze_egcl:
            return
        
        for i in range(self.n_layers):
            layer_name = f"gcl_{i}"
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = True
        
        self.freeze_state = False