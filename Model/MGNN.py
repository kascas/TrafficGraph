import csv
import os
import shutil
import time
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn
from tsnecuda import TSNE


class M_GAT(torch.nn.Module):
    def __init__(self, etypes: list, gnn_dims: list, output_dim: int, num_heads=3, attn_drop=0, feat_drop=0) -> None:
        super().__init__()
        # Embedding layers transform indexes into embedding vectors
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(14, 8), torch.nn.Embedding(5, 3), torch.nn.Embedding(10, 6)])
        self.etypes = etypes
        # gnn layers
        self.gnn_layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                e: dglnn.GATConv(gnn_dims[i - 1] * num_heads, gnn_dims[i], num_heads=num_heads, activation=F.leaky_relu, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop) for e in self.etypes
            }) if i != 1 else
            torch.nn.ModuleDict({
                e: dglnn.GATConv(gnn_dims[0], gnn_dims[1], num_heads=num_heads, activation=F.leaky_relu, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop) for e in self.etypes
            })
            for i in range(1, len(gnn_dims))
        ])
        self.output_layer = torch.nn.ModuleDict({
            e: dglnn.GATConv(gnn_dims[-1] * num_heads, output_dim, num_heads=1, residual=False, activation=None, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop) for e in self.etypes
        })

    def forward(self, graph: dgl.DGLGraph, inputs):
        # TODO add_self_loop is necessary?
        self.subgraphs = {e: graph['conn', e, 'conn'] for e in self.etypes}
        self.inputs = inputs.clone()
        # Embedding layers
        conn_state_emb = self.emb_layers[0](self.inputs[:, 0].type(torch.int))
        proto_emb = self.emb_layers[1](self.inputs[:, 1].type(torch.int))
        service_emb = self.emb_layers[2](self.inputs[:, 2].type(torch.int))
        self.inputs = torch.cat([conn_state_emb, proto_emb, service_emb, self.inputs[:, 3:]], dim=1)
        # GNN layers
        h = {e: self.inputs for e in self.etypes}
        for gnn_layer in self.gnn_layers:
            h = {e: gnn_layer[e](self.subgraphs[e], h[e]).flatten(1) for e in self.etypes}
        h = {e: self.output_layer[e](self.subgraphs[e], h[e]).flatten(1) for e in self.etypes}
        # mean
        h = torch.stack([h[e].flatten(1) for e in self.etypes], dim=1).mean(dim=1)
        return h


class M_GAT_orig(torch.nn.Module):
    def __init__(self, etypes: list, gnn_dims: list, output_dim: int, num_heads=3, attn_drop=0, feat_drop=0) -> None:
        super().__init__()
        # Embedding layers transform indexes into embedding vectors
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(14, 8), torch.nn.Embedding(5, 3), torch.nn.Embedding(10, 6)])
        # use normal HeteroGraphConv to aggregate neighborhood information
        self.gnn_layers = torch.nn.ModuleList([
            dglnn.HeteroGraphConv({
                e: dglnn.GATConv(gnn_dims[i - 1] * num_heads, gnn_dims[i], num_heads=num_heads, activation=F.leaky_relu, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop)for e in etypes
            }, aggregate='mean') if i != 1 else
            dglnn.HeteroGraphConv({
                e: dglnn.GATConv(gnn_dims[0], gnn_dims[1], num_heads=num_heads, activation=F.leaky_relu, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop)for e in etypes
            }, aggregate='mean')
            for i in range(1, len(gnn_dims))
        ])
        self.output_layer = dglnn.HeteroGraphConv({
            e: dglnn.GATConv(gnn_dims[-1] * num_heads, output_dim, num_heads=1, activation=F.leaky_relu, allow_zero_in_degree=True, attn_drop=attn_drop, feat_drop=feat_drop)for e in etypes
        }, aggregate='mean')

    def forward(self, graph, inputs):
        self.inputs = inputs.clone()
        # Embedding layers
        conn_state_emb = self.emb_layers[0](self.inputs[:, 0].type(torch.int))
        proto_emb = self.emb_layers[1](self.inputs[:, 1].type(torch.int))
        service_emb = self.emb_layers[2](self.inputs[:, 2].type(torch.int))
        self.inputs = torch.cat([conn_state_emb, proto_emb, service_emb, self.inputs[:, 3:]], dim=1)

        h = {'conn': self.inputs}
        for gnn_layer in self.gnn_layers:
            h = {'conn': gnn_layer(graph, h)['conn'].flatten(1)}
        h = self.output_layer(graph, h)['conn'].flatten(1)
        return h


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def train(model: torch.nn.Module, train_graph: dgl.DGLGraph, valid_graph: dgl.DGLGraph, test_graph: dgl.DGLGraph, epochs: int, stop_acc: float = -1, lr=0.001):
    print('\r________________________\nStart training...')
    # optimizer of training
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stat_list = []
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    else:
        shutil.rmtree('./saved_model')
        os.makedirs('./saved_model')
    # start training
    for epoch in range(epochs):
        model.train()
        torch.cuda.synchronize()
        start = time.time()
        # backward propagation

        logits = model(train_graph, train_graph.nodes['conn'].data['feat'])
        train_loss = F.cross_entropy(logits, train_graph.nodes['conn'].data['label'])
        # train_loss = FocalLoss(gamma=2)(logits, train_graph.nodes['conn'].data['label'])
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

        print('Epoch {} \ttrain_acc: {:.4f}\ttrain_loss: {:.4f}\tvalid_acc: {:.4f}\tvalid_loss: {:.4f}\ttime: {:.4f}s'.format(epoch + 1, train_acc, train_loss.item(), valid_acc, valid_loss, end - start))

        if stop_acc != -1 and stop_acc <= valid_acc:
            torch.save(model, './model.pt')
            record_train(stat_list)
            return
        if epoch % 100 == 0 and epoch != 0:
            torch.save(model, './saved_model/model_' + str(epoch) + '.pt')
            record_train(stat_list)
            evaluate(model, test_graph, is_test=True)
    torch.save(model, './model.pt')
    record_train(stat_list)
    evaluate(model, test_graph, is_test=True)


def evaluate(model: torch.nn.Module, valid_graph: dgl.DGLGraph, is_test=False):
    model.eval()
    with torch.no_grad():
        logits = model(valid_graph, valid_graph.nodes['conn'].data['feat'])
        loss = F.cross_entropy(logits, valid_graph.nodes['conn'].data['label'])
        # loss = FocalLoss(gamma=2)(logits, valid_graph.nodes['conn'].data['label'])
        labels = valid_graph.nodes['conn'].data['label']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # If testing, return a confusion matrix extra
        if is_test:
            labels, indices = labels.cpu(), indices.cpu()
            mat = confusion_matrix(labels, indices)
            # save cf_mat's csv and svg
            np.savetxt('confusion_matrix.csv', mat, delimiter='\t', fmt='%d')
            mat = mat / mat.sum(axis=1).reshape(-1, 1)
            fig = plt.figure()
            seaborn.heatmap(mat, cmap='Blues', annot=True, fmt='.2f')
            fig.savefig('confusion_matrix.svg', bbox_inches='tight', dpi=100)
            plt.close(fig)
            # create report
            with open('report.txt', 'w')as fp:
                fp.write(classification_report(labels, indices, digits=6))
            return correct.item() * 1.0 / len(labels), loss.item(), mat
        else:
            return correct.item() * 1.0 / len(labels), loss.item()


def record_train(stat_list):
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
    plt.close(fig)


def tsne_visualization(model=None, test_graph=None):
    if model is None:
        model = torch.load('model.pt')
        model = model.cuda()
    if test_graph is None:
        [_, _, test_graph], _ = dgl.load_graphs('./Data/Dataset/train_valid_test.graph')
        test_graph = test_graph.to('cuda:0')
    model.eval()

    def scatter_drawer(logits, labels, file):
        logits = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        print('tsne processing...')
        emb = TSNE(n_iter=350).fit_transform(logits)
        colors = np.array(['black', 'grey', 'red', 'orange', 'olive', 'green', 'lime', 'aqua', 'blue', 'fuchsia', 'purple'])
        fig = plt.figure()
        scatter = plt.scatter(emb[:, 0], emb[:, 1], c=colors[labels], s=1.5)
        # plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        fig.savefig(file, bbox_inches='tight')
        plt.close(fig)

    logits = test_graph.nodes['conn'].data['feat']
    labels = test_graph.nodes['conn'].data['label']
    scatter_drawer(logits, labels, 'emb_input.svg')

    logits = model(test_graph, test_graph.nodes['conn'].data['feat'])
    labels = test_graph.nodes['conn'].data['label']
    scatter_drawer(logits, labels, 'emb_hidden.svg')
