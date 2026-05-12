import math

import numpy as np
from torch.utils.data import Dataset


class DataAgentDataset(Dataset):
    def __init__(self, dataset, ratio=0.5, num_epoch=None, delta=0.875):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.current_epoch = 0

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, targets = self.dataset[index]
        weight = self.weights[index]
        return data, targets, index, weight

    def set_epoch(self, epoch):
        self.current_epoch = epoch + 1

    def prune(self):
        selected_samples = []
        num_samples = len(self.scores)
        num_to_select = int(num_samples * self.ratio)
        selected = np.argsort(self.scores)[-num_to_select:]

        if len(selected) > 0:
            self.weights[selected] = 1 / self.ratio
            selected_samples.extend(selected)

        skipped_samples = len(self.dataset) - len(selected_samples)
        print(f'Skip {skipped_samples} samples for the next iteration')
        self.save_num += skipped_samples
        np.random.shuffle(selected_samples)
        return selected_samples

    def pruning_sampler(self):
        return DataAgentSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return DataAgentSampler(self.no_prune)

    def get_weights(self, indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))


class DataAgentSampler:
    def __init__(self, data_agent_dataset, num_epoch=math.inf, delta=1):
        self.data_agent_dataset = data_agent_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.warmup = 1
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed += 1
        if self.seed > self.stop_prune or self.data_agent_dataset.current_epoch < self.warmup:
            print('Data Agent pruning is not active...')
            if self.seed <= self.stop_prune + 1:
                self.data_agent_dataset.reset_weights()
            self.seq = self.data_agent_dataset.no_prune()
        else:
            self.seq = self.data_agent_dataset.prune()
        self.new_length = len(self.seq)

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        for idx in self.seq:
            yield idx
        self.reset()
