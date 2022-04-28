import csv
import time
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import seaborn

'''
class MSAGE(torch.nn.Module):
    def __init__(self, in_feats: int, out_feats: int, rels: list) -> None:
        super().__init__()
        # self.ndata stores nodes' feature
        self.ndata = None
        # self.trans provides linear transformation in attention mechanism
        # the shape of self.trans is (out_feat, in_feat)
        self.trans = torch.nn.parameter.Parameter(torch.empty(out_feats, in_feats))
        # gnn_layer use GraphSAGE-LSTM to aggregate information within one dimension, then use attention to aggregate information across dimensions
        self.gnn_layer = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, out_feats, aggregator_type='lstm', activation=F.leaky_relu)for rel in rels
        }, aggregate='mean')
        self.init_params()

    def forward(self, graph: dgl.DGLGraph, inputs: dict):
        self.ndata = inputs['conn']
        return self.gnn_layer(graph, inputs)

    def attention(self, tensors: list, dsttype):
        # change tensors's shape to (node_num, dim_num, out_feats)
        tensors = torch.stack(tensors, 1).to('cuda:0')
        # X.shape is (node_num, dim_num, out_feats), q.shape is (node_num, in_feats, 1), self,trans.shape is (out_feats, in_feats)
        X, q = tensors, self.ndata.unsqueeze(2)
        # (node_num, dim_num, out_feats)*(out_feats, in_feats) => (node_num, dim_num, in_feats)
        att_score = torch.matmul(X, self.trans)
        # (node_num, dim_num, in_feats)*(node_num, in_feats, 1) => (node_num, dim_num, 1)
        att_score = torch.matmul(att_score, q)
        att_score = F.softmax(att_score, dim=1)
        # (node_num, out_feats, dim_num)*(node_num, dim_num, 1) => (node_num, out_feats, 1)
        return X.permute(0, 2, 1).matmul(att_score).squeeze(2)

    def init_params(self):
        torch.nn.init.xavier_uniform_(self.trans)
        return
'''


class Model(torch.nn.Module):
    def __init__(self, rels, num_class) -> None:
        super().__init__()
        # Embedding layers transform indexes into embedding vectors
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(14, 6), torch.nn.Embedding(5, 2), torch.nn.Embedding(10, 4)])
        # MGNN layers aggregate neighborhood information (DEPRECATED)
        # self.layer1 = MSAGE(46, 64, rels)
        # self.layer2 = MSAGE(64, 80, rels)
        # self.layer3 = MSAGE(80, 96, rels)
        # use normal HeteroGraphConv to aggregate neighborhood information
        self.layer1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(46, 64, aggregator_type='lstm', activation=F.leaky_relu)for rel in rels
        }, aggregate='mean')
        self.layer2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(64, 80, aggregator_type='lstm', activation=F.leaky_relu)for rel in rels
        }, aggregate='mean')
        self.layer3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(80, 96, aggregator_type='lstm', activation=F.leaky_relu)for rel in rels
        }, aggregate='mean')
        # Linear layer transforms the representation into a vector with fixed dimension
        self.linear1 = torch.nn.Linear(96, 72, bias=True)
        self.linear2 = torch.nn.Linear(72, num_class, bias=True)

    def forward(self, graph, inputs):
        # Embedding layers
        conn_state_emb = self.emb_layers[0](inputs['conn'][:, 0].type(torch.int))
        proto_emb = self.emb_layers[1](inputs['conn'][:, 1].type(torch.int))
        service_emb = self.emb_layers[2](inputs['conn'][:, 2].type(torch.int))
        # h is the actual input of MGNN layer
        h = {'conn': torch.cat([conn_state_emb, proto_emb, service_emb, inputs['conn'][:, 3:]], dim=1)}
        # MGNN layers
        h = self.layer1(graph, h)
        h = self.layer2(graph, h)
        h = self.layer3(graph, h)
        # Linear layers
        h = {k: self.linear1(v) for k, v in h.items()}
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = {k: self.linear2(v) for k, v in h.items()}
        return h


def train(model: torch.nn.Module, train_graph: dgl.DGLGraph, valid_graph: dgl.DGLGraph, epochs: int, stop_acc: float = -1):
    print('\r________________________\nStart training...')
    # optimizer of training
    opt = torch.optim.Adam(model.parameters(), lr=0.0004)
    stat_list = []
    fp = open('train_test.txt', 'w')
    # start training
    for epoch in range(epochs):
        model.train()
        torch.cuda.synchronize()
        start = time.time()
        # backward propagation
        logits = model(train_graph, {'conn': train_graph.nodes['conn'].data['feat']})['conn']
        train_loss = F.cross_entropy(logits, train_graph.nodes['conn'].data['label'])
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        # train_acc and train_loss
        labels = train_graph.nodes['conn'].data['label']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        train_acc = correct.item() * 1.0 / len(labels)
        # valid_acc and valid_loss
        valid_acc, valid_loss = evaluate(model, valid_graph)
        # add train_acc, train_loss, valid_acc and valid_loss to stat_list
        stat_list.append([epoch + 1, train_acc, train_loss.item(), valid_acc, valid_loss])

        torch.cuda.synchronize()
        end = time.time()

        record = '\r=> Epoch {}\ttrain_acc: {:.4f}\ttrain_loss: {:.4f}\tvalid_acc: {:.4f}\tvalid_loss: {:.4f}\ttime: {:.4f}s'.format(epoch + 1, train_acc, train_loss.item(), valid_acc, valid_loss, end - start)
        print(record)
        fp.write(record + '\n')

        if stop_acc != -1 and stop_acc <= valid_acc:
            torch.save(model, 'model.bin')
            break
    fp.close()
    # save the acc and loss of training and validation
    with open('train_test.csv', 'w', newline='') as fp:
        csv.writer(fp).writerows(stat_list)
    # draw acc-loss curve
    stat_list = np.array(stat_list)
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('train_acc')
    plt.plot(stat_list[:, 0], stat_list[:, 1])
    plt.subplot(2, 2, 2)
    plt.title('train_loss')
    plt.plot(stat_list[:, 0], stat_list[:, 2])
    plt.subplot(2, 2, 3)
    plt.title('valid_acc')
    plt.plot(stat_list[:, 0], stat_list[:, 3])
    plt.subplot(2, 2, 4)
    plt.title('valid_loss')
    plt.plot(stat_list[:, 0], stat_list[:, 4])
    fig.savefig('train_test.svg', bbox_inches='tight', dpi=100)


def evaluate(model: torch.nn.Module, valid_graph: dgl.DGLGraph, is_test=False):
    model.eval()
    with torch.no_grad():
        logits = model(valid_graph, {'conn': valid_graph.nodes['conn'].data['feat']})['conn']
        loss = F.cross_entropy(logits, valid_graph.nodes['conn'].data['label'])
        labels = valid_graph.nodes['conn'].data['label']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # If testing, return a confusion matrix extra
        if is_test:
            labels, indices = labels.cpu(), indices.cpu()
            mat = confusion_matrix(labels, indices)
            # save cf_mat's csv and svg
            np.savetxt('confusion_matrix.csv', mat, delimiter='\t', fmt='%.4f')
            mat = mat / mat.sum(axis=1).reshape(-1, 1)
            fig = plt.figure()
            seaborn.heatmap(mat, cmap='Blues', annot=True, fmt='.2f')
            fig.savefig('confusion_matrix.svg', bbox_inches='tight', dpi=100)
            # create report
            with open('report.txt', 'w')as fp:
                fp.write(classification_report(labels, indices, digits=6))
            return correct.item() * 1.0 / len(labels), loss.item(), mat
        else:
            return correct.item() * 1.0 / len(labels), loss.item()
