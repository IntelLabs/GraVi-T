import os
import glob
import torch
from torch_geometric.data import Data, Dataset


class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs):
        super(GraphDataset, self).__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        data = torch.load(self.all_graphs[idx])
        return data
