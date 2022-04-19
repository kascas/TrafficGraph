import os
import json
import random
import networkx as nx
import csv
from tsnecuda import TSNE
import numpy as np
from matplotlib.pyplot import plt

# dataset_initialize() build dataset from labelled summaries
#     sum_dir defines the folder of summaries
#     sum_list defines the order of summaries
#
# dataset_conn_feat() build dataset from connections' features


sum_list = [
    '07-04-ftp_patator.json',
    '07-04-ssh_patator.json',
    '07-05-dos_sloworis.json',
    '07-05-dos_slowhttptest.json',
    '07-05-dos_hulk.json',
    '07-05-dos_goldeneye.json',
    # '07-05-heartbleed.json',
    '07-06-webattack_bruteforce.json',
    # '07-06-webattack_sql.json',
    # '07-06-webattack_xss.json',
    # '07-06-infiltration.json',
    '07-07-botnet.json',
    '07-07-portscan.json',
    '07-07-ddos.json',
]


def dataset_initialize(sum_dir: str = './Data/Summary/Labelled', sum_list: list = sum_list):
    raw_dir = './Data/Dataset/raw'
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    f_train = open('./Data/Dataset/raw/train.json', 'w')
    f_valid = open('./Data/Dataset/raw/valid.json', 'w')
    f_test = open('./Data/Dataset/raw/test.json', 'w')

    train_rate, valid_rate = 0.6, 0.2
    train_dict, valid_dict, test_dict = dict(), dict(), dict()
    train_total, valid_total, test_total = 0, 0, 0

    for file in sum_list:
        if file not in os.listdir(sum_dir):
            raise Exception(file + ' not in ')
        print('>>>', file)
        with open(os.path.join(sum_dir, file), 'r')as fin:
            for line in fin:
                item = json.loads(line)
                rand = random.random()
                if rand <= train_rate:
                    f_train.write(json.dumps(item) + '\n')
                    if item['label'] not in train_dict:
                        train_dict[item['label']] = 0
                    train_dict[item['label']] += 1
                    train_total += 1
                elif rand <= train_rate + valid_rate:
                    f_valid.write(json.dumps(item) + '\n')
                    if item['label'] not in valid_dict:
                        valid_dict[item['label']] = 0
                    valid_dict[item['label']] += 1
                    valid_total += 1
                else:
                    f_test.write(json.dumps(item) + '\n')
                    if item['label'] not in test_dict:
                        test_dict[item['label']] = 0
                    test_dict[item['label']] += 1
                    test_total += 1

    f_train.close()
    f_valid.close()
    f_test.close()

    train_dict = dict(sorted(train_dict.items(), key=lambda x: x[0]))
    valid_dict = dict(sorted(valid_dict.items(), key=lambda x: x[0]))
    test_dict = dict(sorted(test_dict.items(), key=lambda x: x[0]))

    print('-------------------------')
    print('|=>', 'TRAIN\t', train_dict, 'total:', train_total)
    print('|=>', 'VALID\t', valid_dict, 'total:', valid_total)
    print('|=>', 'TEST \t', test_dict, 'total:', test_total)
    print('-------------------------')
    print('|=>', 'ALL  \t', {label: train_dict[label] + valid_dict[label] + test_dict[label] for label in train_dict})
    print('-------------------------')


def build_commgraph(raw_dir: str = './Data/Dataset/raw'):
    for file in os.listdir(raw_dir):
        # node_dict stores every host's id, graph stores the comm_graph, host_dict stores host's stat info
        node_dict, comm_graph, host_list = dict(), nx.DiGraph(), list()
        # files below store different info of comm_graph
        edge_file = os.path.join(raw_dir, '../', os.path.splitext(file)[0] + '.ip.edge')
        node_file = os.path.join(raw_dir, '../', os.path.splitext(file)[0] + '.ip.node')
        graph_file = os.path.join(raw_dir, '../', os.path.splitext(file)[0] + '.ip.gexf')
        host_file = os.path.join(raw_dir, '../', os.path.splitext(file)[0] + '.ip.host')
        with open(os.path.join(raw_dir, file), 'r')as fin:
            print('>>>', os.path.join(raw_dir, file))
            for line in fin:
                item = json.loads(line)
                conn = item['conn']
                sip, dip = conn['id.orig_h'], conn['id.resp_h']
                sport, dport = conn['id.orig_p'], conn['id.resp_p']
                orig_bytes, resp_bytes = conn['orig_bytes'], conn['resp_bytes']
                orig_pkts, resp_pkts = conn['orig_pkts'], conn['resp_pkts']
                # add host to node_dict
                if sip not in node_dict:
                    node_dict[sip] = len(node_dict)
                if dip not in node_dict:
                    node_dict[dip] = len(node_dict)
                # add host to comm_graph
                sid, did = node_dict[sip], node_dict[dip]
                comm_graph.add_node(sid)
                comm_graph.add_node(did)
                comm_graph.add_edge(sid, did)
                # add attrs to corresponding edge in comm_graph
                if 'orig_bytes' not in comm_graph[sid][did]:
                    comm_graph[sid][did]['orig_bytes'] = 0
                    comm_graph[sid][did]['resp_bytes'] = 0
                    comm_graph[sid][did]['orig_pkts'] = 0
                    comm_graph[sid][did]['resp_pkts'] = 0
                comm_graph[sid][did]['orig_bytes'] += orig_bytes
                comm_graph[sid][did]['resp_bytes'] += resp_bytes
                comm_graph[sid][did]['orig_pkts'] += orig_pkts
                comm_graph[sid][did]['resp_pkts'] += resp_pkts
                # add stat info to node in comm_graph
                if 'send_bytes' not in comm_graph.nodes[sid]:
                    comm_graph.nodes[sid]['send_bytes'] = 0
                    comm_graph.nodes[sid]['recv_bytes'] = 0
                    comm_graph.nodes[sid]['send_pkts'] = 0
                    comm_graph.nodes[sid]['recv_pkts'] = 0
                    comm_graph.nodes[sid]['port_stat'] = set()
                comm_graph.nodes[sid]['send_bytes'] += orig_bytes
                comm_graph.nodes[sid]['rec6v_bytes'] += resp_bytes
                comm_graph.nodes[sid]['send_pkts'] += orig_pkts
                comm_graph.nodes[sid]['recv_pkts'] += resp_pkts
                if 'send_bytes' not in comm_graph.nodes[did]:
                    comm_graph.nodes[did]['send_bytes'] = 0
                    comm_graph.nodes[did]['recv_bytes'] = 0
                    comm_graph.nodes[did]['send_pkts'] = 0
                    comm_graph.nodes[did]['recv_pkts'] = 0
                    comm_graph.nodes[did]['port_stat'] = set()
                comm_graph.nodes[did]['send_bytes'] += resp_bytes
                comm_graph.nodes[did]['recv_bytes'] += orig_bytes
                comm_graph.nodes[did]['send_pkts'] += resp_pkts
                comm_graph.nodes[did]['recv_pkts'] += orig_pkts
                comm_graph.nodes[did]['port_stat'].add(dport)
            for nid in range(len(comm_graph.nodes)):
                comm_graph.nodes[nid]['port_stat'] = len(comm_graph.nodes[nid]['port_stat'])
                host_list.append(comm_graph.nodes[nid].values())
            nx.write_edgelist(comm_graph, edge_file, delimiter='\t', data=False)
            nx.write_edgelist(comm_graph, edge_file + '_attr', data=['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts'])
            nx.write_gexf(comm_graph, graph_file)
            # nx.write_edgelist(comm_graph, graph_file, delimiter=',', data=['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts'])

            fnode, fhost = open(node_file, 'w'), open(host_file, 'w', newline='')
            json.dump(node_dict, fnode, indent=2)
            csvwriter = csv.writer(fhost)
            # csvwriter.writerow(['send_bytes', 'recv_bytes', 'send_pkts', 'recv_pkts', 'port_stat'])
            csvwriter.writerows(host_list)
            fnode.close()
            fhost.close()
    return


def tsne_visual(X: np.ndarray):
    emb = TSNE().fit_transform(X)
    x, y = emb[:, 0], emb[:, 1]
    plt.plot(x, y)
    plt.show()


def tsne(X: np.ndarray):
    return TSNE().fit_transform(X)


if __name__ == '__main__':
    # dataset_initialize()
    build_commgraph()
 