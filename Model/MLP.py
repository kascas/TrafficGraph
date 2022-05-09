from sklearn.metrics import classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import time
import numpy as np
import seaborn
import dgl
from tsnecuda import TSNE
import os
import shutil


class MLP(torch.nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(51, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(14, 8), torch.nn.Embedding(5, 3), torch.nn.Embedding(10, 6)])
        self.output = torch.nn.Linear(64, num_class)

    def forward(self, inputs, hidden_only=False):
        # Embedding layers
        conn_state_emb = self.emb_layers[0](inputs[:, 0].type(torch.int))
        proto_emb = self.emb_layers[1](inputs[:, 1].type(torch.int))
        service_emb = self.emb_layers[2](inputs[:, 2].type(torch.int))
        # h is the actual input of MGNN layer
        inputs = torch.cat([conn_state_emb, proto_emb, service_emb, inputs[:, 3:]], dim=1)
        h = self.hidden(inputs)
        if hidden_only:
            return h
        h = self.output(h)
        return h


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
        logits = model(train_graph.nodes['conn'].data['feat'])
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
    tsne_visualization(model, test_graph)


def evaluate(model: torch.nn.Module, valid_graph: dgl.DGLGraph, is_test=False):
    model.eval()
    with torch.no_grad():
        logits = model(valid_graph.nodes['conn'].data['feat'])
        loss = F.cross_entropy(logits, valid_graph.nodes['conn'].data['label'])
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
        emb = TSNE(n_iter=360).fit_transform(logits)
        colors = np.array(['black', 'grey', 'red', 'orange', 'olive', 'green', 'lime', 'aqua', 'blue', 'fuchsia', 'purple'])
        fig = plt.figure()
        scatter = plt.scatter(emb[:, 0], emb[:, 1], c=colors[labels], s=1.5)
        # plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        fig.savefig(file, bbox_inches='tight')
        plt.close(fig)

    logits = test_graph.nodes['conn'].data['feat']
    labels = test_graph.nodes['conn'].data['label']
    scatter_drawer(logits, labels, 'emb_input.svg')

    logits = model(test_graph.nodes['conn'].data['feat'], hidden_only=True)
    labels = test_graph.nodes['conn'].data['label']
    scatter_drawer(logits, labels, 'emb_hidden.svg')

    logits = model(test_graph.nodes['conn'].data['feat'], hidden_only=False)
    labels = test_graph.nodes['conn'].data['label']
    scatter_drawer(logits, labels, 'emb_output.svg')
