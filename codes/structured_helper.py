from imports import *
from pandas.api.types import is_string_dtype, is_numeric_dtype

def convert_cats(train, col_not_convert = [], test = None):
    """
    Convert columns in a training and test dataframe
    into category codes 
    
    train: Training dataframe
    col_not_convert: list of columns which are string type and which you don't want to convert
    test: Test/ Validation set (Mapping from category codes from training)
    """
    
    for n,c in train.items():
        if is_string_dtype(c) and n not in col_not_convert : train[n] = c.astype('category').cat.as_ordered()
    
    if test is not None:
        for n,c in test.items():
            if (n in train.columns) and (train[n].dtype.name=='category'):
                test[n] = pd.Categorical(c, categories=train[n].cat.categories, ordered=True)
            
    cols = list(train.columns[train.dtypes == 'category'])
    for c in cols:
        train[c] = train[c].astype('category').cat.codes
        if test is not None: test[c] = test[c].astype('category').cat.codes

def target_encoder_nonregularized(df_train, df_test, cols, target):
    """
    df_train : dataframe where we want to make changes
    df_test : test df
    cols : cols to make target encoder for
    target : target column (in list)
    
    Returns : dataframe with target encoded columns in df_train itself
    """
    for col in cols:
        gby = df_train.groupby(col).mean()[target]
        dict_gby = dict(zip(list(gby.index), list(gby.values)))
        df_train['%s_mean_%s' % (target,col)] = df_train[col].map(dict_gby)
        m = df_train[col].mean()
        
        df_test['%s_mean_%s' % (target,col)] = df_test[col].map(dict_gby)
        
        df_train[col].fillna(m, inplace=True)
        df_test[col].fillna(m, inplace=True)
        
def target_encoder_regularized(train, test, cols_encode, target, folds = 5):
    """
    Mean and Max regularized target encoding based on kfold
    
    """
    
    kf = KFold(len(train), n_folds=folds, shuffle=True, random_state=1)
    #     set_trace()
    test_copy = test.copy()
    train_copy = train.copy()
    
    for col in cols_encode:       
        train_copy['%s_max_reg_%s' % (target, col)] = np.zeros(len(train_copy))
        train_copy['%s_mean_reg_%s' % (target, col)] = np.zeros(len(train_copy))
        test_copy['%s_max_reg_%s' % (target, col)] = np.zeros(len(test_copy))
        test_copy['%s_mean_reg_%s' % (target, col)] = np.zeros(len(test_copy))
        
        for train_index, val_index in kf:

            max_target = train.loc[train_index, col].map(
            train.loc[train_index].groupby(col).target.max())

            train_copy.loc[val_index,'%s_max_reg_%s' % (col)] = train_copy.loc[val_index, col].map(max_target)

            mean_target = train_copy.loc[train_index, col].map(
            train_copy.loc[train_index].groupby(col).target.mean())

            train_copy.loc[val_index,'%s_mean_reg_%s' % (col)] = train_copy.loc[val_index, col].map(mean_target)


        global_mean = train_copy.target.mean()
        global_max = train_copy.target.max()
        train_copy['%s_max__reg_%s' % (col)].fillna(global_max, inplace=True)
        train_copy['%s_mean_reg_%s' % (col)].fillna(global_mean, inplace=True)
        
    # making test encoding using full training data    
    test_copy['%s_max_reg_%s' % (col)] = test_copy[col].map(train_copy.groupby(col).target.max())
    test_copy['%s_mean_reg_%s' % (col)] = test_copy[col].map(train_copy.groupby(col).target.mean())
            
    return train_copy, test_copy
        
        
def drop_columns(df_train, df_test, cols):
    """
    Drop a list of columns from both train df and test df
    """
    
    df_train.drop(cols, inplace=True, axis=1)
    df_test.drop(cols, inplace=True, axis=1)
    
