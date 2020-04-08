from collections import defaultdict
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import random
from torch.nn.parameter import Parameter
import dgl
import dgl.function as fn

from utils import *


def get_graph(network_data, vocab):
    graphs = []
    num_nodes = len(vocab)

    for edge_type in network_data:
        tmp_data = network_data[edge_type]
        edges = []
        for edge in tmp_data:
            edges.append((vocab[edge[0]].index, vocab[edge[1]].index))
        g = dgl.graph(edges, 'user', edge_type, num_nodes)
        graphs.append(g)
    graph = dgl.hetero_from_relations(graphs)
    
    return graph


# train_pairs: size: 452200, form: (node1, node2, layer_id)
# neighbors: [num_nodes=511, 2, 10]

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class NeighborSampler(object):
    def __init__(self, g, num_fanouts):
        """
        num_fanouts : list of fanouts on each layer.
        """
        self.g = g
        self.num_fanouts = num_fanouts
        
    def sample(self, pairs):
        seeds = []
        seed_pair = []
        seed_type = []
        for pair in pairs:
            if pair[0] not in seeds:
                seeds.append(pair[0])
                seed_pair.append(pair[1])
                seed_type.append(pair[2])
        seeds = torch.LongTensor(seeds)
        # seeds, order = torch.unique(torch.LongTensor(seeds), sorted=True, return_inverse=True)
        blocks = []
        for fanout in reversed(self.num_fanouts):
            sampled_graph = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            sampled_block = dgl.to_block(sampled_graph, seeds)
            seeds = sampled_block.srcdata[dgl.NID]
            blocks.insert(0, sampled_block)
        # print()
        return blocks, torch.LongTensor(seed_pair), torch.LongTensor(seed_type)


class DGLGATNE(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_types, edge_type_count, dim_a):
        super(DGLGATNE, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_types = edge_types
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a

        self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        # edge embedding, 511 * 2 * 10
        self.node_type_embeddings = Parameter(torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size))
        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # embs: [batch_size64, embedding_size200]
    def forward(self, block):
        input_nodes = block.srcdata[dgl.NID]
        output_nodes = block.dstdata[dgl.NID]
        batch_size = block.number_of_dst_nodes('user')
        node_embed = self.node_embeddings  # 511, 200
        node_type_embed = []

        with block.local_scope():
            for i in range(self.edge_type_count):
                edge_type = self.edge_types[i]
                block.nodes['user'].data[edge_type] = self.node_type_embeddings[input_nodes, i]
                block.update_all(fn.copy_u(edge_type, 'm'), fn.sum('m', edge_type), etype=edge_type)
                # print(block.nodes['user'].data[edge_type+'neigh'])
                node_type_embed.append(block.dstnodes['user'].data[edge_type])
        
            node_type_embed = torch.stack(node_type_embed, 1)  # 64, 2, 10
            tmp_node_type_embed = node_type_embed.unsqueeze(2).view(-1, 1, self.embedding_u_size)  # batch*2, 1, 10
            trans_w = self.trans_weights.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(
                -1, self.embedding_u_size, self.embedding_size
            )
            trans_w_s1 = self.trans_weights_s1.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(
                -1, self.embedding_u_size, self.dim_a
            )
            trans_w_s2 = self.trans_weights_s2.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(-1, self.dim_a, 1)

            attention = F.softmax(
                torch.matmul(
                    torch.tanh(torch.matmul(tmp_node_type_embed, trans_w_s1)), trans_w_s2
                ).squeeze(2).view(-1, self.edge_type_count),
                dim=1,
            ).unsqueeze(1).repeat(1, self.edge_type_count, 1)  # 64,edge_type_count2,2
            
            # 64*edge_type_count2, 1, 10
            node_type_embed = torch.matmul(attention, node_type_embed).view(-1, 1, self.embedding_u_size) 
            # 64, 2, 200
            node_embed = node_embed[output_nodes].unsqueeze(1).repeat(1, self.edge_type_count, 1) + \
                torch.matmul(node_type_embed, trans_w).view(-1, self.edge_type_count, self.embedding_size)
            last_node_embed = F.normalize(node_embed, dim=2)
        
            return last_node_embed  # [batch_size64, edge_type_count2, embedding_size200]

    def inference(self, train_inputs, train_types, node_neigh):
        #                batch_size, batch_size, batch_size * 2 * 10
        node_embed = self.node_embeddings[train_inputs]  # batch_size64 * embedding_size200
        # neighbors' neighbors: [batch_size, layers2, neighbor-samples10, layers2, embedding_u_size10]
        node_embed_neighbors = self.node_type_embeddings[node_neigh]
        # [batch_size, 2, neighbor-samples, embedding_u_size]
        node_embed_tmp = torch.cat(
            [
                node_embed_neighbors[:, i, :, i, :].unsqueeze(1)
                for i in range(self.edge_type_count)
            ],
            dim=1,
        )
        # [batch_size, 2, embedding_u_size10]: the sum of each node's all neighbors
        node_type_embed = torch.sum(node_embed_tmp, dim=2)
        # [batch_size, embedding_u_size10, embedding_size200]
        trans_w = self.trans_weights[train_types]
        # [batch_size, embedding_u_size10, dim_a20]
        trans_w_s1 = self.trans_weights_s1[train_types]
        # [batch_size, dim_a20, 1]
        trans_w_s2 = self.trans_weights_s2[train_types]

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)  # [batch_size64, 1, layers2]
        node_type_embed = torch.matmul(attention, node_type_embed)  # [64, 1, embedding_u_size10]
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)  # [64, 200]
        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed  # [batch_size64, embedding_size200]


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes  # 511
        self.num_sampled = num_sampled  # 5
        self.embedding_size = embedding_size  # 200
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))  # 511*200
        # [ (log(i+2) - log(i+1)) / log(num_nodes + 1)]
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # nsloss(data[0].to(device), embs, data[1].to(device))
    def forward(self, input, embs, label):
    # batch_size, batch_size * embedding_size, batch_size
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


# average_auc, average_f1, average_pr = train_model(training_data_by_type)
def train_model(network_data):
    # all_walks: (2, [8640, 4660], 10)
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)
    vocab, index2word = generate_vocab(all_walks)
    # size: 452200, form: (node1, node2, layer_id)
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)  

    edge_types = list(network_data.keys())  # 2

    num_nodes = len(index2word)  # 511
    edge_type_count = len(edge_types)  # 2
    epochs = args.epoch  # 100
    batch_size = args.batch_size  # 64
    embedding_size = args.dimensions  # 200
    embedding_u_size = args.edge_dim  # 10
    u_num = edge_type_count  # 2
    num_sampled = args.negative_samples  # 5
    dim_a = args.att_dim  # 20
    att_head = 1
    neighbor_samples = args.neighbor_samples  # 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # real neghbors (no skip), shape: [num_nodes=511, 2, 10]
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]  
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:  # no neighbor
                neighbors[i][r] = [i] * neighbor_samples  # regard itself as its neighbor
            elif len(neighbors[i][r]) < neighbor_samples:  # randomly repeat neighbors to reach neighbor_samples
                neighbors[i][r].extend(
                    list(
                        np.random.choice(
                            neighbors[i][r],
                            size=neighbor_samples - len(neighbors[i][r]),
                        )
                    )
                )
            elif len(neighbors[i][r]) > neighbor_samples:  # random pick 10 and remove others
                neighbors[i][r] = list(
                    np.random.choice(neighbors[i][r], size=neighbor_samples)
                )

    g = get_graph(network_data, vocab)
    neighbor_sampler = NeighborSampler(g, [neighbor_samples])
    train_dataloader = torch.utils.data.DataLoader(
        train_pairs, batch_size=batch_size, collate_fn=neighbor_sampler.sample, shuffle=True
    )
    model = DGLGATNE(num_nodes, embedding_size, embedding_u_size, edge_types, edge_type_count, dim_a)
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)
    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4
    )

    best_score = 0
    patience = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_pairs)
        # batches = get_batches(train_pairs, neighbors, batch_size)  # 7066 batches
        # data = get_batches(train_pairs, neighbors)

        data_iter = tqdm.tqdm(
            train_dataloader,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, (block, block_pair, block_type) in enumerate(data_iter):
            # batch by batch, 7066 batches in total
            optimizer.zero_grad()
            # data: node1, node2, layer_id, 10 neighbors of node1. dimension: batch_size/batch_size*10
            # embs: [batch_size64, edge_type_count2, embedding_size200]
            embs = model(block[0].to(device))
            block_type = block_type.to(device)
            embs = embs.gather(1, block_type.view(-1, 1, 1).expand(embs.shape[0], 1, embs.shape[2]))[:, 0]
            loss = nsloss(block[0].dstdata[dgl.NID].to(device), embs, block_pair.to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        model.eval()
        # {'1': {}, '2': {}}
        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)  # [i, i]
            train_types = torch.tensor(list(range(edge_type_count))).to(device)  # [0, 1]
            node_neigh = torch.tensor(
                [neighbors[i] for _ in range(edge_type_count)]  # [2, 2, 10]
            ).to(device)

            node_emb = model.inference(train_inputs, train_types, node_neigh) #[2, 200]

            for j in range(edge_type_count):
                final_model[edge_types[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []
        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        print("valid auc:", np.mean(valid_aucs))
        print("valid pr:", np.mean(valid_prs))
        print("valid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)

        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            best_score = cur_score
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping")
                break
    return average_auc, average_f1, average_pr


if __name__ == "__main__":
    # command: python src/main_pytorch.py --input data/example
    args = parse_args()
    file_name = args.input
    print(args)

    training_data_by_type = load_training_data(file_name + "/train.txt")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/test.txt"
    )
    start = time.time()
    average_auc, average_f1, average_pr = train_model(training_data_by_type)
    end = time.time()

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
    print("Training Time", end-start)
