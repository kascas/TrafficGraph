import datetime
import os
import shutil
import json
import getopt
import sys
from zat import zeek_log_reader

# NOTE: This task should run under linux environment
# Run this program: python3 ZeekProcess.py [-t] [-p]
#     -t    delete temp files (stored in ./ZeekLogs)
#     -p    aggregate protocol logs only

optlist, args = getopt.getopt(sys.argv[1:], 'tp')
save_temp, proto_only = False, False
for opt in optlist:
    if opt[0] == '-t':
        save_temp = True
    if opt[0] == '-p':
        proto_only = True


# Generate Zeek logs for traffic
# Zeek logs are stored in ./Data/ZeekLogs
files = os.listdir('./Data/Traffic')
root_dir = os.path.abspath('./Data/ZeekLogs')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

for file in files:
    pwd = os.path.join(root_dir, file).replace('.pcap', '')
    if not os.path.exists(pwd):
        os.makedirs(pwd)
    os.chdir(pwd)
    file = os.path.join('../../Traffic', file)
    print('>>>', file)
    os.system('zeek -C -r ' + file)
    os.chdir(root_dir)

# Use ZAT to parse logs and aggregate them into summaries
# Summaries are stored in ./Data/Summary
os.chdir('../../')
logdir = './Data/ZeekLogs'
if not os.path.exists('./Data/Summary'):
    os.makedirs('./Data/Summary')

for folder in os.listdir(logdir):
    traffic_dict = dict()
    file_dict = dict()
    print('\n------------------------------')
    files = os.listdir(os.path.join(logdir, folder))
    files.remove('conn.log')
    files = ['conn.log'] + files
    for file in files:
        fullname = os.path.join(logdir, folder, file)
        file = file.replace('.log', '')
        reader = zeek_log_reader.ZeekLogReader(fullname)
        logdict = [row for row in reader.readrows()]
        count = 0
        for item in logdict:
            for key in item:
                if isinstance(item[key], datetime.datetime):
                    item[key] = item[key].timestamp()
                if isinstance(item[key], datetime.timedelta):
                    item[key] = item[key].microseconds / 1000
            # conn and all protocols expect dhcp, which uses 'uids'
            if 'uid' in item:
                count += 1
                uid = item['uid']
                if uid not in traffic_dict:
                    traffic_dict[uid] = dict()
                traffic_dict[uid][file] = item
            # Only for dhcp.log
            if 'uids' in item:
                count += 1
                uids = item['uids']
                for uid in uids.split(','):
                    if uid not in traffic_dict:
                        traffic_dict[uid] = dict()
                    traffic_dict[uid][file] = item
            # Only for files.log
            if 'conn_uids' in item and not proto_only:
                count += 1
                uids, fuid = item['conn_uids'], item['fuid']
                uids = uids.replace('[', '').replace(']', '').split(',')
                for uid in uids:
                    if uid not in traffic_dict:
                        traffic_dict[uid] = dict()
                    if fuid not in file_dict:
                        file_dict[fuid] = [uid]
                    else:
                        file_dict[fuid].append(uid)
                    if file not in traffic_dict[uid]:
                        traffic_dict[uid][file] = [item]
                    else:
                        traffic_dict[uid][file].append(item)
            # For events relative to files, namely ocsp, pe, x509
            if 'id' in item and not proto_only:
                count += 1
                fuid = item['id']
                uids = file_dict[fuid]
                for uid in uids:
                    if uid not in traffic_dict:
                        traffic_dict[uid] = dict()
                    if file not in traffic_dict[uid]:
                        traffic_dict[uid][file] = [item]
                    else:
                        traffic_dict[uid][file].append(item)
            print('\r>>>', folder, '->', file, '[', count, '/', len(logdict), ']', end='')
        print()
    print('\n------------------------------\nError Items:\n')
    traffic_items = []
    for item in traffic_dict.items():
        if 'conn' not in item[1]:
            print(item[1])
        else:
            traffic_items.append(item)
    print('------------------------------')
    conn_list = sorted(traffic_items, key=lambda x: x[1]['conn']['ts'])
    with open(os.path.join('./Data/Summary', folder + '.json'), 'w')as fp:
        fp.writelines([json.dumps(item[1]) + '\n' for item in conn_list])

# Delete ./Data/ZeekLogs
if not save_temp:
    print('\nDeleting logs files...')
    shutil.rmtree('./Data/ZeekLogs')
