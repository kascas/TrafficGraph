import json
import os
import getopt
import sys

# NOTE: Run this program: python3 TrafficLabel.py [-t=]
#     -t    type of label. Type could be int or str
#
# Label for CICIDS2017:
# 0     benign
# 1     ftp_patator
# 2     ssh_patator
# 3     dos_sloworis
# 4     dos_slowhttptest
# 5     dos_hulk
# 6     dos_goldeneye
# 7     webattack_bruteforce
# 8     botnet
# 9     portscan
# 10    ddos

optlist, args = getopt.getopt(sys.argv[1:], 't:')
label_type = 'int'
for opt in optlist:
    if opt[0] == '-t' and (opt[1] == 'int' or opt[1] == 'str'):
        label_type = opt[1]


def check_host(conn: dict, ip1: str, ip2: str, port2: int = -1):
    return (ip1 in conn['id.orig_h'] and ip2 in conn['id.resp_h'] and (True if port2 == -1 else conn['id.resp_p'] == port2)) or \
        (ip1 in conn['id.resp_h'] and ip2 in conn['id.orig_h'] and (True if port2 == -1 else conn['id.orig_p'] == port2))


def cicids_2017_label(path='./Data/Summary'):
    files = os.listdir(path)
    if not os.path.exists(os.path.join(path, 'Labelled')):
        os.makedirs(os.path.join(path, 'Labelled'))
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            continue
        print('>>>', os.path.join(path, file))
        fin, fout = open(os.path.join(path, file), 'r'), open(os.path.join(path, 'Labelled', file), 'w')
        for item in fin:
            item = json.loads(item)
            conn = item['conn']
            if conn['ts'] >= 1499170800 and conn['ts'] <= 1499174400:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 21):
                    item['label'] = 1 if label_type == 'int' else 'ftp_patator'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499188200 and conn['ts'] <= 1499191800:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 22):
                    item['label'] = 2 if label_type == 'int' else 'ssh_patator'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499258940 and conn['ts'] <= 1499260200:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 3 if label_type == 'int' else 'dos_sloworis'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499260500 and conn['ts'] <= 1499261700:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 4 if label_type == 'int' else 'dos_slowhttptest'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499262180 and conn['ts'] <= 1499263200:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 5 if label_type == 'int' else 'dos_hulk'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499263800 and conn['ts'] <= 1499264400:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 6 if label_type == 'int' else 'dos_goldeneye'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499343600 and conn['ts'] <= 1499346000:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 7 if label_type == 'int' else 'webattack_bruteforce'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499432640 and conn['ts'] <= 1499443200:
                if check_host(conn, '205.174.165.73', '192.168.10.15') or check_host(conn, '205.174.165.73', '192.168.10.9') or check_host(conn, '205.174.165.73', '192.168.10.14') or check_host(conn, '205.174.165.73', '192.168.10.5') or check_host(conn, '205.174.165.73', '192.168.10.8'):
                    item['label'] = 8 if label_type == 'int' else 'botnet'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499446500 and conn['ts'] <= 1499451780:
                if check_host(conn, '172.16.0.1', '192.168.10.50'):
                    item['label'] = 9 if label_type == 'int' else 'portscan'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            elif conn['ts'] >= 1499453760 and conn['ts'] <= 1499454960:
                if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
                    item['label'] = 10 if label_type == 'int' else 'ddos'
                else:
                    item['label'] = 0 if label_type == 'int' else 'benign'
            # elif conn['ts'] >= 1499346900 and conn['ts'] <= 1499348220:
            #     if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
            #         item['label'] = 11 if label_type=='int' else 'webattack_xss'
            #     else:
            #         item['label'] = 0 if label_type=='int' else 'benign'
            # elif conn['ts'] >= 1499348400 and conn['ts'] <= 1499348640:
            #     if check_host(conn, '172.16.0.1', '192.168.10.50', 80):
            #         item['label'] = 12 if label_type=='int' else 'webattack_sql'
            #     else:
            #         item['label'] = 0 if label_type=='int' else 'benign'
            # elif conn['ts'] >= 1499278320 and conn['ts'] <= 1499279580:
            #     if check_host(conn, '172.16.0.1', '192.168.10.51', 444):
            #         item['label'] = 13 if label_type=='int' else 'heartbleed'
            #     else:
            #         item['label'] = 0 if label_type=='int' else 'benign'
            # elif conn['ts'] >= 1499361540 and conn['ts'] <= 1499366820:
            #     # if check_host(conn, '205.174.165.73', '192.168.10.8') or check_host(conn, '205.174.165.73', '192.168.10.25') or check_host(conn, '192.168.10.8', '192.168.10.'):
            #     #     item['label'] = 'infiltration'
            #     #     fout.write(json.dumps(item) + '\n')
            #     #     continue
            #     pass
            else:
                item['label'] = 0 if label_type == 'int' else 'benign'
            fout.write(json.dumps(item) + '\n')
        fin.close()
        fout.close()


if __name__ == '__main__':
    cicids_2017_label()
