import torch
from torch.utils.data import Dataset
import json
import numpy as np

from torchvision.transforms import Compose
from transforms.augmentation import PadSequence, SelectSequenceCenter, ShuffleSequence
from transforms.multi_input import MultiInput
from transforms import ToFlatTensor
from torch_geometric.data import Data

from st_gcn.graph import Graph

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, mode, train = True,  label_metric = 'RSE_Label'):
        assert mode in ['skeleton', 'silhouette'], 'Mode must be either "skeleton" or "silhouette"'
        
        self.label_metric = label_metric
        self.data = data

        if mode == 'skeleton':
            graph = Graph('coco')
            self.transform = Compose([
                PadSequence(60),
                SelectSequenceCenter(60),
                ShuffleSequence(False),
                MultiInput(graph.connect_joint, graph.center, enabled=True),
                ToFlatTensor()
                ])

        else:
            self.transform = None


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_sample = self.data[str(idx)]['data']
        sequence = []
        for i in range(len(data_sample)):
            keypoints = np.array(data_sample[i], dtype=np.float32)
            sequence.append(torch.tensor(keypoints.reshape(-1, 3)))

        data_sample = Data(
            x = torch.stack(sequence)
        )

        label_sample = self.data[str(idx)]['labels'][self.label_metric]

        if self.transform:
            data_sample = self.transform(data_sample)

        return (data_sample, label_sample)
    
    
if __name__ == '__main__':
    dataset = CustomDataset(mode = 'skeleton',
                            label_metric='RSE_Label')
    
    x, y = dataset[0]
    print("Data: ", x.shape)
    print("Label: ", y)