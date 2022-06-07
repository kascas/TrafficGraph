import Preprocess.Dataset as dataset
import Model.MGNN as gnn
import Model.MLP as mlp
import getopt
import sys
import dgl
import torch


optlist, args = getopt.getopt(sys.argv[1:], 'ird:t')
init_data, rebuild_graph, specify_dataset = False, False, None

dataset_dict = {
    'cicids2017': {
        0: 0.06, 5: 0.08, 9: 0.1, 10: 0.16
    },
}
for opt in optlist:
    if opt[0] == '-i':
        init_data = True
    if opt[0] == '-r':
        rebuild_graph = True
    if opt[0] == '-d':
        if opt[1] in dataset_dict:
            specify_dataset = dataset_dict[opt[1]]
    if opt[0] == '-t':
        model = torch.load('model.pt')
        model = model.cuda()
        [_, _, test_graph], _ = dgl.load_graphs('./Data/Dataset/train_valid_test.graph')
        test_graph = test_graph.to('cuda:0')
        gnn.evaluate(model, test_graph, is_test=True)
        gnn.tsne_visualization(model, test_graph)
        # mlp.evaluate(model, test_graph, is_test=True)
        # mlp.tsne_visualization(model, test_graph)
        exit(0)

train_graph, valid_graph, test_graph = None, None, None
if init_data:
    dataset.dataset_initialize()
    if specify_dataset != None:
        dataset.dataset_adjust(specify_dataset)
if rebuild_graph:
    train_graph = dataset.build_relation_graph('./Data/Dataset/raw/train.json')
    valid_graph = dataset.build_relation_graph('./Data/Dataset/raw/valid.json')
    test_graph = dataset.build_relation_graph('./Data/Dataset/raw/test.json')

    dgl.save_graphs('./Data/Dataset/train_valid_test.graph', [train_graph, valid_graph, test_graph])
else:
    [train_graph, valid_graph, test_graph], _ = dgl.load_graphs('./Data/Dataset/train_valid_test.graph')
    train_graph = train_graph.to('cuda:0')
    valid_graph = valid_graph.to('cuda:0')
    test_graph = test_graph.to('cuda:0')

dataset.dataset_info()

model = gnn.M_GAT(train_graph.etypes, [64, 72, 72, 72, 32], 12, num_heads=3, attn_drop=0.005, feat_drop=0.03)
# model = mlp.MLP(64, [72, 72, 72, 32], 12)

model = model.cuda()
gnn.train(model, train_graph, valid_graph, test_graph, 50000, lr=0.01, lr_step=1000, lr_gamma=0.8)
_, _, conf_mat = gnn.evaluate(model, test_graph, is_test=True)
# mlp.train(model, train_graph, valid_graph, test_graph, 20000, lr=0.001)
# _, _, conf_mat = mlp.evaluate(model, test_graph, is_test=True)
print(conf_mat)
