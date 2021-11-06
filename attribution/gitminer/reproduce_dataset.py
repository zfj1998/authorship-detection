'''
In：
1. parsed_result_{fold}_{split}.csv
√2. change_metadata.csv
√3. method_ids.csv
4. out_code/creations/{change_id}_{nan}.after.java
Out:
1. author_name;
2. committerName;
3. code content;
4. code file path;
5. change_id;
6. method_id;
7. method_file_name; # out_code中文件的路径
'''
import os
import pickle
import json
import pandas as pd
import numpy as np
import ipdb

BASE_PATH = './out/spring-boot/'
CACHE_PATH = BASE_PATH + 'cache/'

def dump_cache(content, path):
    with open(path, 'wb') as f:
        pickle.dump(content, f)

def load_cache(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def handle_method_ids():
    '''
    返回dict, key为method_id, value为method_name
    '''
    cached_method_ids = CACHE_PATH + 'method_ids.pkl'
    cached_ids = load_cache(cached_method_ids)
    if cached_ids:
        return cached_ids
    file_path = BASE_PATH + 'method_ids.csv'
    method_ids = pd.read_csv(file_path, sep=';')
    m_ids_set = dict()
    for i, row in method_ids.iterrows():
        m_ids_set[row['id']] = row['methodName']
    dump_cache(m_ids_set, cached_method_ids)
    return m_ids_set

def handle_metadata():
    '''
    返回dict, key为change_id, value有author_name, committerName, newPath
    '''
    cached_meta_path = CACHE_PATH + 'change_metadata.pkl'
    cached_meta = load_cache(cached_meta_path)
    if cached_meta:
        return cached_meta
    file_path = BASE_PATH + 'change_metadata.csv'
    metadata = pd.read_csv(file_path)
    meta_dict = dict()
    for i, row in metadata.iterrows():
        meta_dict[row['id']] = {
            'authorName': row['authorName'],
            'committerName': row['committerName'],
            'newPath': row['newPath']
        }
    dump_cache(meta_dict, cached_meta_path)
    return meta_dict

def handle_out_code():
    '''
    返回dict, key为chagne_id, value是该change包含的(file_name, method_code)s
    '''
    cached_outcode_path = CACHE_PATH + 'out_code.pkl'
    cached_outcode = load_cache(cached_outcode_path)
    if cached_outcode:
        return cached_outcode
    code_path = BASE_PATH + 'out_code/creations/'
    code_dict = dict()
    for file_name in os.listdir(code_path):
        with open(code_path + file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        change_id = file_name.replace('.after.java', '').split('_')[0]
        if change_id in code_dict:
            code_dict[change_id].append((file_name, content))
        else:
            code_dict[change_id] = [(file_name, content)]
    dump_cache(code_dict, cached_outcode_path)
    return code_dict

def handle_each_fold(fold, split, method_ids, meta_data, out_code):
    '''
    把每个fold扩展为新的table, 以csv格式存储, 分隔符为|||
    '''
    fold_file_path = f'{BASE_PATH}out_fold_data/parsed_result_{fold}_{split}.csv'
    fold_content = pd.read_csv(fold_file_path)
    csv_dict = {
        'change_id': [], # √
        'author_id': [], # √
        'author_name': [], # √
        'committer_name': [], # √
        'method_id': [], # √
        'method_name': [], # √
        'method_content': [], # √
        'method_file': [], # √
        'method_content_file': [] # √
    }
    for i, row in fold_content.iterrows():
        change_id = row['change_id']
        csv_dict['change_id'].append(change_id)
        csv_dict['author_id'].append(row['label'])
        csv_dict['author_name'].append(row['author_name'])
        committer_name = meta_data[change_id]['committerName']
        csv_dict['committer_name'].append(committer_name)
        method_id = row['method_id']
        csv_dict['method_id'].append(method_id)
        method_name = method_ids[method_id]
        csv_dict['method_name'].append(method_name)

        method_content = None
        method_content_file = None
        candidate_methods = out_code[str(change_id)]
        for f_name, m_content in candidate_methods:
            if method_name not in m_content:
                continue
            method_content = m_content
            method_content_file = f_name
            break
        csv_dict['method_content'].append(method_content)
        csv_dict['method_content_file'].append(method_content_file)
        csv_dict['method_file'].append(meta_data[change_id]['newPath'])
    df = pd.DataFrame(csv_dict)
    out_path = f'{BASE_PATH}/out_fold_data/expanded_{fold}_{split}.csv'
    df.to_csv(out_path, sep='；')
    tmp = pd.read_csv(out_path, sep='；')
    ipdb.set_trace()
    print()


def main():
    method_ids = handle_method_ids()
    meta_data = handle_metadata()
    out_code = handle_out_code()
    for fold in range(1):
        for split in ['test']:
            handle_each_fold(fold, split, method_ids, meta_data, out_code)

if __name__ == '__main__':
    # tmp = handle_method_ids()
    main()
