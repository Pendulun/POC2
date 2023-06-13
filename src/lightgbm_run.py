import lightgbm as lgb
import numpy as np
import pathlib
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def config_parser():
    parser = argparse.ArgumentParser(description='LGB')
    # parser.add_argument('--tf',
    #                     dest='target_folder',
    #                     action='store',
    #                     required=True,
    #                     help="The folder where to save the best model")
    
    parser.add_argument('--ep',
                        dest='embs_path',
                        action='store',
                        required=True,
                        help="The path to the numpy embeddings file")
    
    # parser.add_argument('--model',
    #                     dest='model_name',
    #                     action='store',
    #                     required=False,
    #                     help="The model file name without extension")
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    embeddings = np.load(pathlib.Path(args.embs_path))
    # print(embeddings.shape)

    num_instances = embeddings.shape[0]
    y = [1]*(num_instances//2) + [0]*(num_instances//2)
    # y = [1]*150 + [0]*150
    # print(y[0], y[150], len(y))

    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, 
                                                        test_size=0.2)
    # print(sum(np.array(y_train) == 1), sum(np.array(y_test) == 1))
    # print(sum(np.array(y_train) == 0), sum(np.array(y_test) == 0))

    d_train = lgb.Dataset(X_train, label=y_train)

    grid_params = {
        'learning_rate': (0.1, 0.05, 0.03, 0.01, 0.001),
        'boosting_type': ('gbdt', 'dart', 'goss'),#'rf'),
        'objective': ['binary'],
        'metric':(['auc', 'binary_loss']),
        'num_leaves':(100, 200, 300),
        'max_depth': (50, 100, 200, 300, -1),
        'subsample_for_bin': (10, 20, 30, 50),
        'n_estimators': (50, 100, 200, 300),
    }

    # lgbm_params = {'learning_rate':0.01, 'boosting_type':'gbdt',
    #                'objective':'binary',
    #                'metric':['auc', 'binary_loss'],
    #                'num_leaves':200, 
    #                'max_depth':-1,
    #                'n_estimators':90}
    # epochs = 5000
    
    lgb_model = lgb.LGBMClassifier()
    clf = GridSearchCV(lgb_model, grid_params)
    clf.fit(X_train, y_train)

    # scoring = ['auc', 'binary_loss']
    # scores = cross_validate(clf, X_train, y_train, cv=5, n_jobs=-1, scoring=['f1', 'accuracy', 'roc_auc'])
    # print(scores)
    print(f"best_params: {clf.best_params_}")
    print(f"best_score: {clf.best_score_}")

    # clf.fit(X_train, y_train)
    # # clf = lgb.cv(lgbm_params, d_train, epochs, nfold=5)

    # # print(clf)
    # # print(type(clf))

    # y_pred = clf.predict(X_test)
    # print(y_pred)
    # # print(y_test)
    # # print(len(y_test), sum(y_pred == y_test))

    # for i in range(X_test.shape[0]):
    #     if y_pred[i]>= 0.5:
    #         y_pred[i] = 1
    #     else:
    #         y_pred[i] = 0

    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True)
    # plt.show()

    # print("Acc: ", accuracy_score(y_test, y_pred))
    # print("AUC: ", roc_auc_score(y_test, y_pred))