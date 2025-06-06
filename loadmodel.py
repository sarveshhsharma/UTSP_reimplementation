import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
from utils import TSPLoss,edge_overlap,get_heat_map
import pickle
from torch.utils.data import  Dataset,DataLoader# use pytorch dataloader
from random import shuffle
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=100, help='Graph Size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--use_smoo', action='store_true')
parser.add_argument('--EPOCHS', type=int, default=300,
                    help='epochs to train')
parser.add_argument('--topk', type=int, default=20,
                    help='top k elements per row, should equal to int Rec_Num = 20 in Search/code/include/TSP_IO.h')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=2.,
                    help='temperature for adj matrix')
parser.add_argument('--diag_penalty', type=float, default=3.,
                    help='penalty on the diag')
parser.add_argument('--rescale', type=float, default=1.,
                    help='rescale for xy plane')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
device = args.device


tsp_instances = np.load('UTSP_Mine/UTSP/data/test_tsp_instance_%d.npy'%args.num_of_nodes) # 128 instances
NumofTestSample = tsp_instances.shape[0]
Std = np.std(tsp_instances, axis=1)
Mean = np.mean(tsp_instances, axis=1)


tsp_instances = tsp_instances - Mean.reshape((NumofTestSample,1,2))

tsp_instances = args.rescale * tsp_instances # 2.0 is the rescale

tsp_sols = np.load('UTSP_Mine/UTSP/data/test_tsp_sol_%d.npy'%args.num_of_nodes)
total_samples = tsp_instances.shape[0]
import json

from models import GATModel
#scattering model
model = GATModel(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
from scipy.spatial import distance_matrix


### count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total number of parameters:')
print(count_parameters(model))



def coord_to_adj(coord_arr):
    dis_mat = distance_matrix(coord_arr,coord_arr)
    return dis_mat


tsp_instances_adj = np.zeros((total_samples,args.num_of_nodes,args.num_of_nodes))
for i in range(total_samples):
    tsp_instances_adj[i] = coord_to_adj(tsp_instances[i])
class TSP_Dataset(Dataset):
    def __init__(self, coord,data, targets):
        self.coord = torch.FloatTensor(coord)
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        xy_pos = self.coord[index]
        x = self.data[index]
        y = self.targets[index]
#        tsp_instance = Data(coord=x,sol=y)
        return tuple(zip(xy_pos,x,y))

    def __len__(self):
        return len(self.data)

dataset = TSP_Dataset(tsp_instances,tsp_instances_adj,tsp_sols)
testdata = dataset[0:] ##this is very important!
TestData_size = len(testdata)
batch_size = args.batch_size
test_loader = DataLoader(testdata, batch_size, shuffle=False)
mask = torch.ones(args.num_of_nodes, args.num_of_nodes)
mask.fill_diagonal_(0)

#just a random test
for batch in test_loader:
    xy = batch[0]      # city coordinates [batch_size, num_nodes, 2]
    distance_m = batch[1]   # distance matrix [batch_size, num_nodes, num_nodes]
    sol = batch[2]          # solution tour or labels
print(xy.shape)
print(distance_m.shape)
print(sol.shape)

def test(loader,topk = 20):
    avg_size = 0
    total_cost = 0.0
    full_edge_overlap_count = 0

    TestData_size = len(loader.dataset)
    Saved_indices = np.zeros((TestData_size,args.num_of_nodes,topk))
    Saved_Values = np.zeros((TestData_size,args.num_of_nodes,topk))
    Saved_sol = np.zeros((TestData_size,args.num_of_nodes+1))
    Saved_pos = np.zeros((TestData_size,args.num_of_nodes,2))
    count = 0
    model.eval()
    for batch in loader:
        xy_pos = batch[0]
        distance_m = batch[1]
        sol = batch[2]
        adj = torch.exp(-1.*distance_m/args.temperature)
        adj *= mask
        # start here:
        t0 = time.time()
        output = []
        for i in range(xy_pos.size(0)):      
            output_i = (model(xy_pos[0],adj[0]))
            output.append(output_i)
            output_tensor = torch.stack(output) 
        t1 = time.time()
        Heat_mat = get_heat_map(SctOutput=output_tensor,num_of_nodes=args.num_of_nodes)
        print('It takes %.5f seconds from instance: %d to %d'%(t1 - t0,count,count + batch_size))
        sol_indicies = torch.topk(Heat_mat,topk,dim=2).indices
        sol_values = torch.topk(Heat_mat,topk,dim=2).values
#        print(sol_values.size())
#        print(batch_size)
        Saved_indices[count:batch_size+count] = sol_indicies.detach().cpu().numpy()
        Saved_Values[count:batch_size+count] = sol_values.detach().cpu().numpy()
        Saved_sol[count:batch_size+count] = sol.detach().cpu().numpy()
        Saved_pos[count:batch_size+count] = xy_pos.detach().cpu().numpy()
        count = count + batch_size


    return Saved_indices,Saved_Values,Saved_sol,Saved_pos,Heat_mat


# #TSP200
# model_name = 'UTSP_Mine/UTSP/Saved_Models/TSP_%d/scatgnn_layer_%d_hid_%d_model_5_temp_3.500.pth'%(args.num_of_nodes,args.nlayers,args.hidden)# topk = 10
model_name = 'UTSP_Mine/UTSP/Saved_Models/TSP_100/scatgnn_layer_3_hid_64_model_1_temp_3.500.pth'
model.load_state_dict(torch.load(model_name))
# #Saved_indices,Saved_Values,Saved_sol,Saved_pos = test(test_loader,topk = 8) # epoch=20>10 
Saved_indices,Saved_Values,Saved_sol,Saved_pos,Heat_mat = test(test_loader,topk = args.topk) # epoch=20>10

# print('Finish Inference!')

# import os, sys

Q = Saved_pos
A = Saved_sol 
C = Saved_indices
V = Saved_Values
# # with open("1kTraning_TSP%dInstance_%d.txt"%(args.num_of_nodes,Saved_indices.shape[0]), "w") as f:
# #     for i in range(Q.shape[0]):
# #         for j in range(Q.shape[1]):
# #             f.write(str(Q[i][j][0]) + " " + str(Q[i][j][1]) + " ")
# #         f.write("output ")
# #         for j in range(A.shape[1]):
# #             f.write(str(int(A[i][j] + 1)) + " ")
# #         f.write("indices ")
# #         for j in range(C.shape[1]):
# #             for k in range(args.topk):
# #                 if C[i][j][k] == j:
# #                     f.write("-1" + " ")
# #                 else:
# #                     f.write(str(int(C[i][j][k] + 1)) + " ")
# #         f.write("value ")
# #         for j in range(V.shape[1]):
# #             for k in range(args.topk):
# #                 f.write(str(V[i][j][k]) + " ")
# #         f.write("\n")
# #         if i == Saved_indices.shape[0] - 1:
# #             break


import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_tour_length(tour, coords):
    return sum(euclidean_distance(coords[int(tour[i])], coords[int(tour[(i + 1) % len(tour)])]) for i in range(len(tour)))

def build_greedy_tour(pred_neighbors, coords):
    num_nodes = coords.shape[0]
    visited = [False] * num_nodes
    tour = [0]
    visited[0] = True

    for _ in range(num_nodes - 1):
        last = tour[-1]
        found = False
        for nxt in pred_neighbors[int(last)]:  # Try top-k
            if not visited[int(nxt)]:
                tour.append(nxt)
                visited[int(nxt)] = True
                found = True
                break
        if not found:
            # If no predicted unvisited neighbor is left, pick a random unvisited node
            for j in range(num_nodes):
                if not visited[j]:
                    tour.append(j)
                    visited[j] = True
                    break
    return tour

def two_opt(tour, coords):
    improved = True
    best_distance = compute_tour_length(tour, coords)
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue  # Skip adjacent nodes
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = compute_tour_length(new_tour, coords)
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return tour

# Q = Saved_pos, C = Saved_indices
# Example for instance 0
instance_id = 0
coords = Q[instance_id]  # Shape: (num_nodes, 2)
pred_neighbors = C[instance_id]  # Shape: (num_nodes, topk)

# Step 1: Build tour from top-k neighbors
init_tour = build_greedy_tour(pred_neighbors, coords)
init_tour = [int(node) for node in init_tour]

# Step 2: Refine tour using 2-opt
opt_tour = two_opt(init_tour, coords)

# Print or return the tour
print("Heat MAP: ",Heat_mat)
print("Predicted tour:", opt_tour)
print("Length:", compute_tour_length(opt_tour, coords))