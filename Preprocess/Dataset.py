import math
import os
import json
import random
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


def dataset_initialize(sum_dir: str = './Data/Summary/Labelled', sum_list: list = sum_list, train_scale: int = 0.6, valid_scale: int = 0.2, total_scale: int = 1):
    raw_dir = './Data/Dataset/raw'
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    f_train = open('./Data/Dataset/raw/train.json', 'w')
    f_valid = open('./Data/Dataset/raw/valid.json', 'w')
    f_test = open('./Data/Dataset/raw/test.json', 'w')

    train_scale, valid_scale, test_scale = train_scale * total_scale, valid_scale * total_scale, (1 - train_scale - valid_scale) * total_scale
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
                if item['conn']['orig_pkts'] == 0:
                    continue
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
                elif rand <= train_scale + valid_scale + test_scale:
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


cicids2017 = {
    0: 0.04, 3: 0.1, 6: 0.08, 7: 0.1
}


def dataset_adjust(class_scale: dict = cicids2017):
    files = ['./Data/Dataset/raw/train.json', './Data/Dataset/raw/valid.json', './Data/Dataset/raw/test.json']
    temp = './Data/Dataset/raw/temp.json'

    for file in files:
        stat_dict, total = dict(), 0
        fin, fout = open(file, 'r'), open(temp, 'w')
        for line in fin:
            rand = random.random()
            label = json.loads(line)['label']
            if label not in class_scale or rand < class_scale[label]:
                fout.write(line)
                if label not in stat_dict:
                    stat_dict[label] = 0
                stat_dict[label] += 1
                total += 1
        print('|=>', file + '\t', stat_dict, 'total:', total)
        fin.close()
        fout.close()
        os.remove(file)
        os.rename(temp, file)


def dataset_info():
    f_train = open('./Data/Dataset/raw/train.json', 'r')
    f_valid = open('./Data/Dataset/raw/valid.json', 'r')
    f_test = open('./Data/Dataset/raw/test.json', 'r')

    def get_stat(fp):
        d = dict()
        for line in fp:
            item = json.loads(line)
            if item['label'] not in d:
                d[item['label']] = 0
            else:
                d[item['label']] += 1
        return d

    train_dict = dict(sorted(get_stat(f_train).items(), key=lambda x: x[0]))
    valid_dict = dict(sorted(get_stat(f_valid).items(), key=lambda x: x[0]))
    test_dict = dict(sorted(get_stat(f_test).items(), key=lambda x: x[0]))

    f_train.close()
    f_valid.close()
    f_test.close()

    print(train_dict)
    print(valid_dict)
    print(test_dict)

    with open('dataset_info.txt', 'w')as fp:
        fp.write('> train_data: {}\n'.format(train_dict))
        fp.write('> valid_data: {}\n'.format(valid_dict))
        fp.write('> test_data : {}\n'.format(test_dict))
    return


def build_relation_graph(raw_data: str):
    fp = open(raw_data, 'r')
    rel_latest, rel_edges, count = dict(), dict(), 0
    conn_feat, conn_label = list(), list()

    for rel in ['conn_sdp', 'auth_http', 'auth_ftp', 'auth_ssh', 'state_normal', 'state_abnormal']:
        rel_latest[rel] = dict()
    for rel in ['conn', 'auth', 'state']:
        rel_edges[rel] = [[], []]

    for line in fp:
        item = json.loads(line)
        conn = item['conn']
        sip, dip, dport = conn['id.orig_h'], conn['id.resp_h'], conn['id.resp_p']
        # conn - conn_sdp
        if sip in rel_latest['conn_sdp'] and dip in rel_latest['conn_sdp'][sip] and dport in rel_latest['conn_sdp'][sip][dip]:
            latest_count = rel_latest['conn_sdp'][sip][dip][dport]
            rel_edges['conn'][0] += [latest_count, count]
            rel_edges['conn'][1] += [count, latest_count]
        if sip not in rel_latest['conn_sdp']:
            rel_latest['conn_sdp'][sip] = dict()
        if dip not in rel_latest['conn_sdp'][sip]:
            rel_latest['conn_sdp'][sip][dip] = dict()
        rel_latest['conn_sdp'][sip][dip][dport] = count
        # auth - auth_ftp
        if sip in rel_latest['auth_ftp'] and dip in rel_latest['auth_ftp'][sip]:
            if conn['service'] == 1 and 'ftp' not in item:
                latest_count = rel_latest['auth_ftp'][sip][dip]
                rel_edges['auth'][0] += [latest_count, count]
                rel_edges['auth'][1] += [count, latest_count]
        if sip not in rel_latest['auth_ftp']:
            rel_latest['auth_ftp'][sip] = dict()
        rel_latest['auth_ftp'][sip][dip] = count
        # auth - auth_ssh
        if sip in rel_latest['auth_ssh'] and dip in rel_latest['auth_ssh'][sip]:
            if conn['service'] == 6 and 'ssh' in item and item['ssh']['auth_success'] == False:
                latest_count = rel_latest['auth_ssh'][sip][dip]
                rel_edges['auth'][0] += [latest_count, count]
                rel_edges['auth'][1] += [count, latest_count]
        if sip not in rel_latest['auth_ssh']:
            rel_latest['auth_ssh'][sip] = dict()
        rel_latest['auth_ssh'][sip][dip] = count
        # auth - auth_http
        def check_login(domain, url):
            word_list = ['passport', 'login', 'signin']
            for word in word_list:
                if word in domain or word in url:
                    return True
        if sip in rel_latest['auth_http'] and dip in rel_latest['auth_http'][sip]:
            if conn['service'] == 2 and 'http' in item and check_login(item['http']['host'], item['http']['uri']):
                latest_count = rel_latest['auth_http'][sip][dip]
                rel_edges['auth'][0] += [latest_count, count]
                rel_edges['auth'][1] += [count, latest_count]
        if sip not in rel_latest['auth_http']:
            rel_latest['auth_http'][sip] = dict()
        rel_latest['auth_http'][sip][dip] = count
        # state - state_normal
        if sip in rel_latest['state_normal'] and dip in rel_latest['state_normal'][sip]:
            if conn['conn_state'] == 2:
                latest_count = rel_latest['state_normal'][sip][dip]
                rel_edges['state'][0] += [latest_count, count]
                rel_edges['state'][1] += [count, latest_count]
        if sip not in rel_latest['state_normal']:
            rel_latest['state_normal'][sip] = dict()
        rel_latest['state_normal'][sip][dip] = count
        # state - state_abnormal
        if sip in rel_latest['state_abnormal'] and dip in rel_latest['state_abnormal'][sip]:
            if conn['conn_state'] != 2:
                latest_count = rel_latest['state_abnormal'][sip][dip]
                rel_edges['state'][0] += [latest_count, count]
                rel_edges['state'][1] += [count, latest_count]
        if sip not in rel_latest['state_abnormal']:
            rel_latest['state_abnormal'][sip] = dict()
        rel_latest['state_abnormal'][sip][dip] = count
        # store features and labels
        conn_feat.append(get_conn_feat(conn))
        conn_label.append(item['label'])
        count += 1
        print('\rLoading...', count, end='')
    graph = dgl.heterograph({('conn', rel, 'conn'): (torch.tensor(rel_edges[rel][0]), torch.tensor(rel_edges[rel][1])) for rel in rel_edges}, num_nodes_dict={'conn': count})
    graph.nodes['conn'].data['feat'] = torch.tensor(conn_feat)
    graph.nodes['conn'].data['label'] = torch.tensor(conn_label)
    return graph.to('cuda:0')


def get_conn_feat(conn: dict):
    duration = conn['duration'] if conn['duration'] != 0 else -1
    orig_bytes, resp_bytes = conn['orig_ip_bytes'], conn['resp_ip_bytes']
    orig_pkts, resp_pkts = conn['orig_pkts'], conn['resp_pkts']
    conn_state, history, proto, service = conn['conn_state'], conn['history'], conn['proto'], conn['service']
    orig_pkt_ps, resp_pkt_ps = orig_pkts / duration, resp_pkts / duration
    orig_ip_bytes_ps, resp_ip_bytes_ps = orig_bytes / duration, resp_bytes / duration
    bytes_ratio, pkts_ratio = resp_bytes / orig_bytes if orig_bytes != 0 else -1, resp_pkts / orig_pkts
    return [conn_state, proto, service, *history, duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, orig_ip_bytes_ps, resp_ip_bytes_ps, orig_pkt_ps, resp_pkt_ps, bytes_ratio, pkts_ratio]
