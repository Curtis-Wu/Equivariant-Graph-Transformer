"""
This module holds the implementation of E(n) Equivariant Graph Convolutional Layer
Reference: EGNN: V. G. Satorras et al., https://arxiv.org/abs/2102.09844 
"""

import torch
import torch.nn as nn
from copy import deepcopy
from torch_cluster import radius_graph
from torch_scatter import scatter


class E_GCL(nn.Module):

    def __init__(
        self, input_nf, output_nf, hidden_nf, add_edge_feats=0, act_fn=nn.SiLU(), 
        residual=True, attention=False, normalize=False, coords_agg='mean', static_coord = True
    ):
        '''
        :param intput_nf: Number of input node features
        :param output_nf: Number of output node features
        :param hidden_nf: Number of hidden node features
        :param add_edge_feats: Number of additional edge feature
        :param act_fn: Activation function
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
        :param coords_agg: aggregation function
        '''
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.static_coord = static_coord
        self.coords_agg = coords_agg
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
        # Initializes layer weights using xavier uniform initialization
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # Update coodinates
        if not static_coord:
            # coordinates mlp sequntial layers
            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
        
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

    def forward(self, h, edge_index, coord, add_edge_feats=None, node_attr=None):
        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # calculate radial distances for each pair of node
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # Compute edge features
        edges = self.edge_model(h[row], h[col], radial, add_edge_feats)

        # Update coordinates
        if not self.static_coord:
            coord = self.coord_model(coord, edge_index, coord_diff, add_edge_feats)
            
        # Update node features
        h, agg = self.node_model(h, edge_index, edges, node_attr)

        return h, add_edge_feats
    

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