from sklearn.metrics import classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import time
import numpy as np
import seaborn
import dgl


class MLP(torch.nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(46, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 72),
            torch.nn.ReLU(),
            torch.nn.Linear(72, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, num_class),
        )
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(14, 6), torch.nn.Embedding(5, 2), torch.nn.Embedding(10, 4)])

    def forward(self, inputs):
        # Embedding layers
        conn_state_emb = self.emb_layers[0](inputs[:, 0].type(torch.int))
        proto_emb = self.emb_layers[1](inputs[:, 1].type(torch.int))
        service_emb = self.emb_layers[2](inputs[:, 2].type(torch.int))
        # h is the actual input of MGNN layer
        inputs = torch.cat([conn_state_emb, proto_emb, service_emb, inputs[:, 3:]], dim=1)
        return self.sequential(inputs)


def train(model: torch.nn.Module, train_graph: dgl.DGLGraph, valid_graph: dgl.DGLGraph, epochs: int):
    print('\r________________________\nStart training...')
    # optimizer of training
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    stat_list = []
    fp = open('train_test.txt', 'w')
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

        record = '\r=> Epoch {}\ttrain_acc: {:.4f}\ttrain_loss: {:.4f}\tvalid_acc: {:.4f}\tvalid_loss: {:.4f}\ttime: {:.4f}s'.format(epoch + 1, train_acc, train_loss.item(), valid_acc, valid_loss, end - start)
        print(record)
        fp.write(record + '\n')
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
            np.savetxt('confusion_matrix.csv', mat, delimiter='\t', fmt='%.2f')
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
