from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np


class StateActionReturnDataset(Dataset):
    def __init__(self, data: List[dict], context_length):
        self.data = data
        self.block_size = context_length
        self.vocab_size = 8
        self.problem_size = 200

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, instance_id):
        instance_step_number = self.data[instance_id].get("step_number")
        # if instance_step_number == 0:
        #     instance_id += 1
        #     instance_step_number += 1

        if instance_step_number + self.block_size > (self.problem_size):
            instance_id -= (instance_step_number + self.block_size - self.problem_size)

        observation_sequence_array = np.array([instance.get("observation")
                                               for instance in self.data[instance_id:instance_id+self.block_size]])
        action_sequence_array = np.array([instance.get("action")
                                          for instance in self.data[instance_id:instance_id+self.block_size]])
        target_sequence_array = np.array([instance.get("action")
                                          for instance in self.data[instance_id:instance_id+self.block_size]])
        return_to_go_sequence_array = np.array([instance.get("returns_to_go")
                                                for instance in self.data[instance_id:instance_id+self.block_size]])
        rewards = np.array([instance.get("reward")
                                                for instance in self.data[instance_id:instance_id + self.block_size]])
        action_mask_sequence_array = np.array([instance.get("action_mask")
                                               for instance in self.data[instance_id:instance_id+self.block_size]])

        #TODO: ist das unsqueeze notwendig
        states = torch.tensor(observation_sequence_array, dtype=torch.float32).reshape(self.block_size, -1) #(block_size, 57)
        actions = torch.tensor(action_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        #returns_to_go = torch.tensor(return_to_go_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        returns_to_go = torch.tensor(return_to_go_sequence_array, dtype=torch.float64).unsqueeze(1) # (block_size, 1) #NOTE for neuro ls we need float rtgs
        timesteps = torch.tensor([self.data[instance_id].get("step_number")], dtype=torch.int64).unsqueeze(1) #(block_size, 1)
            targets = torch.tensor(target_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        #action_masks = torch.tensor(action_mask_sequence_array, dtype=torch.bool).unsqueeze(1)

        return states, actions, returns_to_go, timesteps, targets#, action_masks
