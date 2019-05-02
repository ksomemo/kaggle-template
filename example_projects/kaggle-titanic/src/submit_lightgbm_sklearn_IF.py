import numpy as np
import pandas as pd
from tqdm import tqdm
from logging import StreamHandler, DEBUG, INFO, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, accuracy_score
import lightgbm as lgb


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


def get_logger(module_name, filepath, level=DEBUG):
    logger = getLogger(module_name)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(filepath, mode='a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    return logger


def main():
    """
    https://github.com/tkm2261/kaggle-youtube-porto/blob/master/protos/train.py
    https://www.kaggle.com/currypurin/simple-lightgbm
    https://www.kaggle.com/currypurin/tutorial-of-kaggle-ver3-hurokub
    """
    logger = get_logger(__name__, 'log.log')
    logger.info('start')
    
    logger.info('load data start')
    train = pd.read_csv('../input/train.csv')
    x_test = pd.read_csv('../input/test.csv')
    x_train = train.drop('Survived', axis=1)
    y_train = train.loc[:, ['Survived']]
    logger.info('load data end')

    # Sexの変換
    logger.info('preprocessing start')
    genders = {'female': 0, 'male':1}
    x_train['Sex'] = x_train['Sex'].map(genders)
    x_test['Sex'] = x_test['Sex'].map(genders)
    
    # Embarkedの変換
    embarked = {'S':0, 'C':1, 'Q':2}
    x_train['Embarked'] = x_train['Embarked'].map(embarked)
    x_test['Embarked'] = x_test['Embarked'].map(embarked)
    
    # 不要な列の削除
    submission = x_test.loc[:, ['PassengerId']]
    x_train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    x_test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    logger.info('preprocessing end')

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {
        'max_depth': range(3, 6+1),
        'reg_alpha': [1],
        'reg_lambda': [0],
        'importance_type': ['gain', 'split'],
        'random_state': [0]
    }
    min_score = float('inf')
    min_params = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params:', params)

        accuracy = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            trn_y = y_train.loc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]
            val_y = y_train.loc[valid_idx, :]

            clf = lgb.LGBMClassifier(objective='binary', **params)
            clf.fit(trn_x, trn_y,
                    eval_set=[(val_x, val_y)], eval_metric='logloss',
                    categorical_feature=['Sex', 'Embarked'],
                    early_stopping_rounds=50, verbose=5)
            pred = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
            pred = (pred > 0.5).astype(int)
            
            acc = accuracy_score(val_y, pred)
            accuracy.append(acc)
            logger.debug(f'\taccuracy: {acc}')

        sc_accuracy = np.mean(accuracy)
        if min_score > sc_accuracy:
            min_score = sc_accuracy
            min_params = params
        logger.info(f'accuracy: {sc_accuracy}')
        logger.info('current min score: {min_score}, params: {min_params}')

    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum gini: {}'.format(min_score))

    clf = lgb.LGBMClassifier(objective='binary', **min_params)
    clf.fit(x_train, y_train,
            categorical_feature=['Sex', 'Embarked'],
            verbose=5)

    logger.info('test data load end {}'.format(x_test.shape))
    pred_test = clf.predict(x_test)

    submission['Survived'] = pred_test    
    submission.to_csv('submit_lightgbm_sklearn_IF.csv', index=False)

    logger.info('end')


if __name__ == '__main__':
    main()
