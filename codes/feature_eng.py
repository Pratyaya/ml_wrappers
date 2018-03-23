from imports import *

def model_validation(alg, X, Y,  verbose=True, scoring = metrics.roc_auc_score, *args, **kwargs):
    """
    K-fold model validation. Use it when you have small training data or no good validation set.
    X = training feature df
    Y = training label df
    scoring = scorer we want to track. default is AUC
    
    Returns: Kfold mean score of given model 
    """
    
    score_mean = 0
    kf = KFold(len(X),
               n_folds = 5,
               shuffle = True,
               random_state = 1)
    fold = 1
    for train_index, val_index in kf:
        
        x_train, y_train = X.loc[train_index], Y.loc[train_index]
        x_val, y_val = X.loc[val_index], Y.loc[val_index]
        
        alg.fit(x_train, y_train)
        fold_score = scoring(y_val, alg.predict_proba(x_val)[:,1], *args)

        if verbose == 'All' : print('Fold val score. Fold : %s : ' % (fold), fold_score)
        fold += 1
        score_mean += fold_score
    score_mean = score_mean/kf.n_folds
    
    if verbose is True or verbose == 'All' : print('Mean score : ', score_mean)
    print('\n')
    return score_mean

def feat_selection(alg, X, Y, scoring = metrics.roc_auc_score, col_list = None, thres = 0., *args):
    """
    A different way to do feature selection
    Fits model with all columns except one, one at a time. This helps to find features which making model worse when removed. 
    
    Returns: 
    col_score:
    cols_keep: Columns to keep based on threshold difference cause by removing that column
    """
    
    col_score = {}
    print('Fitting model with all columns')
    col_score['all_cols'] = model_validation(alg, X, Y, verbose=False, scoring)
    cols_keep = []
    
    if col_list is None:
        for col in X.columns:
            print('Fitting model without : ' , col)
            X2 = X.drop(col, axis=1)
            col_score[col] = model_validation(alg, X2, Y, verbose=False, scoring)
            if col_score[col] <= col_score['all_cols'] + thres : # minimum diff 0.001
                cols_keep.append(col)   
    
    else:
        for col in col_list:
            X2 = X.drop(col, axis=1)
            col_score[col] = model_validation_auc(alg, X2, Y, verbose=False, scoring)
            if col_score[col] <= col_score['all_cols'] + thres  : # minimum diff 0.001
                cols_keep.append(col)
                
    return col_score, cols_keep
