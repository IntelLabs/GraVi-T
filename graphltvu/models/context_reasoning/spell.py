import torch
from torch.nn import Module, Sequential, ReLU, Dropout
from torch_geometric.nn import Linear, EdgeConv, SAGEConv, BatchNorm


class SPELL(Module):
    def __init__(self, cfg):
        super(SPELL, self).__init__()
        self.num_modality = cfg['num_modality']
        channels = [cfg['channel1'], cfg['channel2']]
        proj_dim = cfg['proj_dim']
        final_dim = cfg['final_dim']
        dropout = cfg['dropout']

        self.layerspf = Linear(-1, proj_dim) # projection layer for spatial features
        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        self.layer21 = SAGEConv(channels[0], channels[1])
        self.batch21 = BatchNorm(channels[1])

        self.layer31 = SAGEConv(channels[1], final_dim)
        self.layer32 = SAGEConv(channels[1], final_dim)
        self.layer33 = SAGEConv(channels[1], final_dim)


    def forward(self, x, c, edge_index, edge_attr):
        feature_dim = x.shape[1]
        x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layerspf(c)), dim=1))
        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:])
            x = x_visual + x_audio

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        return x1+x2+x3
