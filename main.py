import Preprocess.Dataset as dataset
import Model.MGNN as gnn


if __name__=='__main__':
    dataset.dataset_initialize(total_scale=0.1)
    train_graph = dataset.build_relation_graph('./Data/Dataset/raw/train.json')
    valid_graph = dataset.build_relation_graph('./Data/Dataset/raw/valid.json')
    test_graph = dataset.build_relation_graph('./Data/Dataset/raw/test.json')

    model = gnn.Model(train_graph.etypes)
    model = model.cuda()

    gnn.train(model, train_graph, valid_graph, 1000)

    _, _, conf_mat = gnn.evaluate(model, test_graph, has_confusion_matrix=True)
    print(conf_mat)