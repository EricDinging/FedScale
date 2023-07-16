import sys
[sys.path.append(i) for i in ['..']]

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import fedscale.cloud.config_parser as parser
from fedscale.dataloaders.femnist import FEMNIST
from fedscale.dataloaders.utils_data import get_data_transform
from fedscale.dataloaders.divide_data import DataPartitioner

train_transform, test_transform = get_data_transform('mnist')
train_dataset = FEMNIST('../benchmark/dataset/data/femnist', dataset='train', transform=train_transform)
test_dataset = FEMNIST('../benchmark/dataset/data/femnist', dataset='test', transform=test_transform)

# partition the dataset
parser.args.task = 'cv'
training_sets = DataPartitioner(data=train_dataset, args=parser.args, numOfClass=62)
training_sets.partition_data_helper(num_clients=None,
                                    data_map_file='../benchmark/dataset/data/femnist/client_data_mapping/train.csv')

# testing_sets = DataPartitioner(data=test_dataset, args=parser.args, numOfClass=62, isTest=True)
# testing_sets.partition_data_helper(num_clients=None, data_map_file='./benchmark/dataset/data/femnist/client_data_mapping/test.csv')

print(type(training_sets))
print(f'Total number of data samples: {training_sets.getDataLen()}')
print(f'Total number of clients: {training_sets.getClientLen()}')

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
size_dist = training_sets.getSize()['size']

n_bins = 50
axs[0].hist(size_dist, bins=n_bins)
axs[0].set_title('Client data size distribution')

label_dict = training_sets.getClientLabel()
axs[1].hist(label_dict, bins=n_bins)
axs[1].set_title('Client label distribution')

plt.show()

rank=1
isTest = False
dropLast = True
partition = training_sets.use(rank-1, isTest)
num_loaders = min(int(len(partition)/parser.args.batch_size/2), parser.args.num_loaders)
dataloader = DataLoader(partition, batch_size=16, shuffle=True
                        , pin_memory=True, timeout=60, num_workers=num_loaders, drop_last=dropLast)

for data in iter(dataloader):
    plt.imshow(np.transpose(data[0][0].numpy(), (1, 2, 0)))
    break

