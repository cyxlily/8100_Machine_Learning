import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]

def normalize_data(X):
    features = X.columns.tolist()
    new_df = pd.DataFrame(columns = features)
    for f in features:
        current = X[f].tolist()
        min_v = np.min(current)
        max_v = np.max(current)
        delta = max_v - min_v
        if delta != 0:
            temp = (current - min_v) / delta
        else:
            temp = current
        new_df[f] = temp
    return new_df

def split_dataset(df, test_size):
    train, test = train_test_split(df, test_size = test_size)
    return train, test
    
    
def norm_split(dataset, label, file_name, test_size, full_size, nfeatures, column_format, seporator, header, missing, drops, seperated):
    os.chdir('{}/data'.format(home))
    
    try:        
        if seporator != 0:
            df = pd.read_csv(file_name, sep = seporator, header = header)
            if df.shape[1] > nfeatures + 1:                
                for f in df.columns.tolist():
                    try:
                        if np.isnan(df[f].tolist()).all():
                            df = df.drop(columns = [f])
                    except TypeError:
                        pass            
        else:
            df = pd.read_excel(file_name, header = header)
    except IOError:
        print("cannot read file {}".format(dataset))

    if len(drops) > 0:
        df = df.drop(columns = drops)    
    
    if missing == np.nan:
        df = df.dropna().reset_index(drop=True)
    else:
        droplist = []
        for i in range(df.shape[0]):
            current = df.iloc[i].tolist()
            if missing in current:
                droplist.append(i)
        if len(droplist) > 0:
            df = df.drop(index = droplist).reset_index(drop=True)
    
    if label == -1:
        label = df.columns.tolist()[-1]
    Y = df[label].tolist()
    X = df.drop(columns = [label])
    
    
    
    if column_format != 0:
        for i in range(X.shape[0]):
            for j in X.columns.tolist():
                current = X.at[i, j]
                try:
                    index = current.index(':')
                    X.at[i, j] = current[index+1:]
                except ValueError:
                    continue
    
    X = X.astype('float')
    X = normalize_data(X)
    df = X
    df['label'] = Y
    
    df = df.dropna().reset_index(drop=True)
    if df.shape[0] != full_size:
        print('size not equal')
        exit()
    if df.shape[1] != nfeatures + 1:
        print('nfeatures not equal')
        exit()
        
    if seperated == True:   
        index = file_name.index('.')
        name = file_name[:index]     
        os.chdir('{}/data'.format(home))
        try:
            os.mkdir(dataset)
        except OSError:
            pass
        os.chdir('{}/data/{}'.format(home, dataset))
        df.to_pickle('{}_df.pkl'.format(name))
        return
        
        
        
    train, test = split_dataset(df, test_size)
    train = train.sort_index()
    test = test.sort_index()
    os.chdir('{}/data'.format(home))
    try:
        os.mkdir(dataset)
    except OSError:
        pass
    os.chdir('{}/data/{}'.format(home, dataset))
    train.to_pickle('{}_train_df.pkl'.format(dataset))
    test.to_pickle('{}_test_df.pkl'.format(dataset))
    
    
    
    
    
