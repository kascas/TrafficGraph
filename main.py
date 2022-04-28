import pickle
import Preprocess.Dataset as dataset
import Model.MGNN as gnn
import Model.MLP as mlp
import getopt
import sys
import dgl


if __name__ == '__main__':
    optlist, args = getopt.getopt(sys.argv[1:], 'ird:')
    init_data, rebuild_graph, specify_dataset = False, False, None

    dataset_dict = {
        'cicids2017': {
            0: 0.04, 5: 0.08, 9: 0.08, 10: 0.15
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

    model = gnn.Model(train_graph.etypes, 11)
    model = model.cuda()
    gnn.train(model, train_graph, valid_graph, 5000)
    _, _, conf_mat = gnn.evaluate(model, test_graph, is_test=True)
    print(conf_mat)

    # model = mlp.MLP(11)
    # model = model.cuda()
    # mlp.train(model, train_graph, valid_graph, 5000)
    # _, _, conf_mat = mlp.evaluate(model, test_graph, is_test=True)
    # print(conf_mat)
