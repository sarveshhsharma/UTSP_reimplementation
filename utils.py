import numpy as np
import torch

def TSPLoss(SctOutput,distance_matrix,num_of_nodes):
    '''
    input:
    SctOutput: batchsize * num_of_nodes * num_of_nodes tensor
    distance_matrix: batchsize * num_of_nodes * num_of_nodes tensor
    '''
    HeatM = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 1, 2),-1, 1))
    weighted_path = torch.mul(HeatM, distance_matrix)
    weighted_path = weighted_path.sum(dim=(1,2))
    return weighted_path, HeatM

def get_heat_map(SctOutput,num_of_nodes):
    '''
    input:
    SctOutput: batchsize * num_of_nodes * num_of_nodes tensor
    '''
    HeatM = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 1, 2),-1, 1))
    return HeatM


