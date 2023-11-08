from copy import deepcopy
import torch.nn as nn
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    Reference: EGNN: V. G. Satorras et al., https://arxiv.org/abs/2102.09844 
    """
    
    def __init__(
        self, input_nf, output_nf, hidden_nf, add_edge_feats=0, act_fn=nn.SiLU(), 
        residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False
    ):
        '''
        :param intput_nf: Number of input node features
        :param output_nf: Number of output node features
        :param hidden_nf: Number of hidden node features
        :param add_edge_feats: Number of edge feature
        :param act_fn: Activation function
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param coords_agg: aggregation function
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        # Number of features used to describe the relative positions between nodes
        # Because we're using radial distance, so dimension = 1
        edge_coords_nf = 1
        # input_edge stores the node values, one edge connects two nodes, so * 2
        input_edge = input_nf * 2
        # Prevents division by zeroes, numerical stability purpose
        self.epsilon = 1e-8
        
        # mlp operation for edges
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + add_edge_feats, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        # mlp operation for nodes
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        # Initializes layer weights using uniform initialization
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # coordinates mlp sequntial layers
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # attention mlp layer
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_feats):
        # concatenation of edge features
        if edge_feats is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            # Dimension analysis:
            # eg. source, target -> (num_edges, num_node_features)
            # radial -> (num_edges, 1)
            # edge_feats -> (num_edges, 3)
            # out -> (num_edges, num_node_features*2 + 1 + 3)
            out = torch.cat([source, target, radial, edge_feats], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_feats, node_attr = None):
        # Dimension analysis:
        # x -> (num_nodes, num_node_features)
        # edge_index -> (2, num_edges)
        # edge_feats -> (num_edges, num_edge_feats)

        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # unsorted_segment_sum sums up all edge features for each node
        # agg dimension -> (num_nodes, num_edge_feats)
        agg = unsorted_segment_sum(edge_feats, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # calculate coordinate difference between node
        coord_diff = coord[row] - coord[col]
        # calculate the radial distance for each pair of node
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        # normalization
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_feats=None, node_attr=None):
        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # calculate radial distances for each pair of node
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # Compute edge features
        edge_feat = self.edge_model(h[row], h[col], radial, edge_feats)
        # Update coordinates
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        # Update node features
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_feats


class EGNN(nn.Module):
    def __init__(self, 
        hidden_channels, num_edge_feats=0, act_fn=nn.SiLU(), n_layers=4, 
        residual=True, attention=False, normalize=False, tanh=False, 
        max_atom_type=100, cutoff=5.0, max_num_neighbors=32, **kwargs
    ):
        '''
        :param max_atom_type: Number of features for 'h' at the input
        :param hidden_channels: Number of hidden features
        :param num_edge_feats: Number of additional edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.max_atom_type = max_atom_type
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        # Create embeddings of dimension (hidden_channels, ) for each atom type
        self.type_embedding = nn.Embedding(max_atom_type, hidden_channels)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(
                input_nf = self.hidden_channels, 
                output_nf = self.hidden_channels, 
                hidden_nf = self.hidden_channels, 
                add_edge_feats = num_edge_feats,
                act_fn=act_fn, residual=residual, 
                attention=attention, normalize=normalize, tanh=tanh))
        
        # Output energy from the last layer, which is a vector of dimension (hidden_channels, )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(), 
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, z, pos, batch, edge_index=None, edge_feats=None):
        # Retrieve embedding for each node(atom) type
        # Dimension: (num_nodes, num_feats), so for 5 nodes with 256 features -> (5,256)
        h = self.type_embedding(z)
        # Copy coordinates to x
        # Dimension: (num_nodes, dimension), so for 5 nodes with 3 dimenions -> (5,3)
        x = deepcopy(pos)
        # If edge_index was not provided
        # Dimension: (2, num_edges), so for a graph with 10 edges -> (2,10)
        if edge_index is None:
            # Calculates edge_index from graph structure based on cutoff radius
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors + 1,
            )
        # Loop over all the Equivariant graph convolutional layers
        # To update node embeddings and coordinates
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edge_index, x, edge_feats=edge_feats)
        # Aggregate node features 'h' across the batch using summation
        # h have dimension of ()
        out = scatter(h, batch, dim=0, reduce='add')
        # Outputs energy from the graph level representation
        out = self.energy_head(out)
        return out, x - pos


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
