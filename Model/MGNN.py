import time
import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys


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
        }, aggregate=self.attention)
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


class Model(torch.nn.Module):
    def __init__(self, in_feats, out_feats, rels) -> None:
        super().__init__()
        self.layer1 = MSAGE(in_feats, 12, rels)
        self.layer2 = MSAGE(12, 16, rels)
        self.layer3 = MSAGE(16, 20, rels)
        self.linear = torch.nn.Linear(20, out_feats, bias=True)

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = self.layer2(graph, h)
        h = self.layer3(graph, h)
        return {k: self.linear(v) for k, v in h.items()}


def train(model: torch.nn.Module, train_graph: dgl.DGLGraph, valid_graph: dgl.DGLGraph, epochs: int):
    print('\r________________________\nStart training...')
    opt = torch.optim.Adam(model.parameters())
    x_list, train_acc_list, train_loss_list, valid_acc_list, valid_loss_list = [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        torch.cuda.synchronize()
        start = time.time()

        logits = model(train_graph, {'conn': train_graph.nodes['conn'].data['feat']})['conn']
        train_loss = F.cross_entropy(logits, train_graph.nodes['conn'].data['label'])
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        labels = train_graph.nodes['conn'].data['label']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        train_acc = correct.item() * 1.0 / len(labels)
        valid_acc, valid_loss = evaluate(model, valid_graph)

        x_list.append(epoch + 1)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        torch.cuda.synchronize()
        end = time.time()
        print('\r=> Epoch {}\ttrain_acc: {:.4f}\ttrain_loss: {:.4f}\tvalid_acc: {:.4f}\tvalid_loss: {:.4f}\ttime: {:.4f}s'.format(epoch + 1, train_acc, train_loss.item(), valid_acc, valid_loss, end - start))
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('train_acc')
    plt.plot(x_list, train_acc_list)
    plt.subplot(2, 2, 2)
    plt.title('train_loss')
    plt.plot(x_list, train_loss_list)
    plt.subplot(2, 2, 3)
    plt.title('valid_acc')
    plt.plot(x_list, valid_acc_list)
    plt.subplot(2, 2, 4)
    plt.title('valid_loss')
    plt.plot(x_list, valid_loss_list)
    plt.show()
    fig.savefig('train_test.svg')


def evaluate(model: torch.nn.Module, valid_graph: dgl.DGLGraph, has_confusion_matrix=False):
    model.eval()
    with torch.no_grad():
        logits = model(valid_graph, {'conn': valid_graph.nodes['conn'].data['feat']})['conn']
        loss = F.cross_entropy(logits, valid_graph.nodes['conn'].data['label'])
        labels = valid_graph.nodes['conn'].data['label']
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        if has_confusion_matrix:
            return correct.item() * 1.0 / len(labels), loss.item(), confusion_matrix(labels.cpu(), indices.cpu())
        else:
            return correct.item() * 1.0 / len(labels), loss.item()


if __name__ == '__main__':
    sys.path.append('./')
    from Preprocess.Dataset import build_relation_graph

    train_graph = build_relation_graph('./Data/Dataset/raw/train.json')
    valid_graph = build_relation_graph('./Data/Dataset/raw/valid.json')
    test_graph = build_relation_graph('./Data/Dataset/raw/test.json')

    model = Model(7, 11, train_graph.etypes)
    model = model.cuda()

    train(model, train_graph, valid_graph, 10000)

    _, _, conf_mat = evaluate(model, test_graph, has_confusion_matrix=True)
    print(conf_mat)
