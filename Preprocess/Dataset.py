import os
import json
import random
import networkx as nx
import csv
from tsnecuda import TSNE
import numpy as np
import matplotlib.pyplot as plt
import dgl
import torch


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


def dataset_initialize(sum_dir: str = './Data/Summary/Labelled', sum_list: list = sum_list, train_scale:int = 0.6, valid_scale:int = 0.2, total_scale:int = 1):
    raw_dir = './Data/Dataset/raw'
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    f_train = open('./Data/Dataset/raw/train.json', 'w')
    f_valid = open('./Data/Dataset/raw/valid.json', 'w')
    f_test = open('./Data/Dataset/raw/test.json', 'w')

    train_scale, valid_scale,test_scale = train_scale * total_scale, valid_scale * total_scale, (1 - train_scale - valid_scale) * total_scale
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
                if rand <= train_scale:
                    f_train.write(line)
                    if item['label'] not in train_dict:
                        train_dict[item['label']] = 0
                    train_dict[item['label']] += 1
                    train_total += 1
                elif rand <= train_scale + valid_scale:
                    f_valid.write(line)
                    if item['label'] not in valid_dict:
                        valid_dict[item['label']] = 0
                    valid_dict[item['label']] += 1
                    valid_total += 1
                elif rand<=train_scale + valid_scale+test_scale:
                    f_test.write(line)
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
    print('|=>', 'ALL  \t', {label: train_dict[label] + valid_dict[label] + test_dict[label] for label in train_dict}, 'total:', train_total + valid_total + test_total)
    print('-------------------------')
    return

cicids2017={
    0: 0.04, 3: 0.1, 6: 0.08, 7: 0.1
}

def dataset_adjust(class_scale:dict=cicids2017):
    files=['./Data/Dataset/raw/train.json','./Data/Dataset/raw/valid.json','./Data/Dataset/raw/test.json']
    temp='./Data/Dataset/raw/temp.json'

    for file in files:
        stat_dict, total=dict(),0
        fin,fout=open(file,'r'),open(temp,'w')
        for line in fin:
            rand=random.random()
            label=json.loads(line)['label']
            if label not in class_scale or rand < class_scale[label]:
                fout.write(line)
                if label not in stat_dict:
                    stat_dict[label]=0
                stat_dict[label]+=1
                total+=1
        print('|=>', file+'\t', stat_dict, 'total:', total)
        fin.close()
        fout.close()
        os.remove(file)
        os.rename(temp,file)


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


def build_relation_graph(raw_data:str):
    fp = open(raw_data, 'r')
    rel_latest, rel_edges, count = dict(), dict(), 0
    conn_feat, conn_label = list(), list()

    for rel in ['conn_sd', 'conn_s', 'conn_d', 'ftp_login', 'ssh_login', 'http_login']:
        rel_latest[rel] = dict()
        rel_edges[rel] = ([], [])

    for line in fp:
        item = json.loads(line)
        conn = item['conn']
        sip, dip = conn['id.orig_h'], conn['id.resp_h']
        # 'conn_sd' relation
        if sip in rel_latest['conn_sd'] and isinstance(rel_latest['conn_sd'][sip],dict) and dip in rel_latest['conn_sd'][sip]:
            rel_edges['conn_sd'][0].append(rel_latest['conn_sd'][sip][dip])
            rel_edges['conn_sd'][1].append(count)
            rel_edges['conn_sd'][1].append(rel_latest['conn_sd'][sip][dip])
            rel_edges['conn_sd'][0].append(count)
        else:
            rel_latest['conn_sd'][sip] = dict()
        rel_latest['conn_sd'][sip][dip] = count
        # 'conn_s' relation
        if sip in rel_latest['conn_s']:
            rel_edges['conn_s'][0].append(rel_latest['conn_s'][sip])
            rel_edges['conn_s'][1].append(count)
            rel_edges['conn_s'][1].append(rel_latest['conn_s'][sip])
            rel_edges['conn_s'][0].append(count)
        rel_latest['conn_s'][sip]=count
        # 'conn_d' relation
        if dip in rel_latest['conn_d']:
            rel_edges['conn_d'][0].append(rel_latest['conn_d'][dip])
            rel_edges['conn_d'][1].append(count)
            rel_edges['conn_d'][1].append(rel_latest['conn_d'][dip])
            rel_edges['conn_d'][0].append(count)
        rel_latest['conn_d'][dip]=count
        # 'ftp_login' relation
        if dip in rel_latest['ftp_login'] and conn['service']==1 and 'ftp' not in item:
            rel_edges['ftp_login'][0].append(rel_latest['ftp_login'][dip])
            rel_edges['ftp_login'][1].append(count)
            rel_edges['ftp_login'][1].append(rel_latest['ftp_login'][dip])
            rel_edges['ftp_login'][0].append(count)
        rel_latest['ftp_login'][dip]=count
        # 'ssh_login' relation
        if dip in rel_latest['ssh_login'] and conn['service']==6 and 'ssh' in item and item['ssh']['auth_success']==False:
            rel_edges['ssh_login'][0].append(rel_latest['ssh_login'][dip])
            rel_edges['ssh_login'][1].append(count)
            rel_edges['ssh_login'][1].append(rel_latest['ssh_login'][dip])
            rel_edges['ssh_login'][0].append(count)
        rel_latest['ssh_login'][dip]=count
        # 'http_login' relation
        def check_login(domain,url):
            word_list=['passport','login','signin']
            for word in word_list:
                if word in domain or word in url:
                    return True
        if dip in rel_latest['http_login'] and conn['service']==2 and 'http' in item and check_login(item['http']['host'],item['http']['uri']):
            rel_edges['http_login'][0].append(rel_latest['http_login'][dip])
            rel_edges['http_login'][1].append(count)
            rel_edges['http_login'][1].append(rel_latest['http_login'][dip])
            rel_edges['http_login'][0].append(count)
        rel_latest['http_login'][dip]=count
        # store features and labels
        conn_feat.append(get_conn_feat(conn))
        conn_label.append(item['label'])
        count += 1
        print('\rLoading...',count,end='')
    graph = dgl.heterograph({('conn',rel,'conn'):(torch.tensor(rel_edges[rel][0]), torch.tensor(rel_edges[rel][1])) for rel in rel_edges})
    graph.nodes['conn'].data['feat']=torch.tensor(conn_feat)
    graph.nodes['conn'].data['label']=torch.tensor(conn_label)
    return graph.to('cuda:0')

def get_conn_feat(conn:dict):
    duration = conn['duration']
    orig_bytes, resp_bytes = conn['orig_bytes'], conn['resp_bytes']
    orig_ip_bytes, resp_ip_bytes = conn['orig_ip_bytes'], conn['resp_ip_bytes']
    orig_pkts, resp_pkts = conn['orig_pkts'], conn['resp_pkts']
    conn_state, history, proto, service = conn['conn_state'], conn['history'], conn['proto'], conn['service']
    orig_pkt_interval, resp_pkt_interval = orig_pkts / duration if duration != 0 else 0, resp_pkts / duration if duration != 0 else 0
    return [conn_state, proto, service, *history, duration, orig_bytes, resp_bytes, orig_ip_bytes, resp_ip_bytes, orig_pkts, resp_pkts, orig_pkt_interval, resp_pkt_interval]

def tsne_visual(X: np.ndarray):
    emb = TSNE().fit_transform(X)
    x, y = emb[:, 0], emb[:, 1]
    fig = plt.figure()
    plt.plot(x, y)
    plt.show()
    fig.savefig('emb.svg')


def tsne(X: np.ndarray):
    return TSNE().fit_transform(X)
 