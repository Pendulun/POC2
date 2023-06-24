import argparse
import pathlib
import pickle
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

def config_parser():
    parser = argparse.ArgumentParser(description='LGB')
    
    parser.add_argument('--ep',
                        dest='embs_path',
                        action='store',
                        required=True,
                        help="The path to the numpy embeddings file")

    parser.add_argument('--rf',
                        dest='results_folder',
                        action='store',
                        required=True,
                        help="The path to the folder where to save the results")
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    random_state = 42

    embeddings = np.load(pathlib.Path(args.embs_path))
    num_instances = embeddings.shape[0]
    y = [1]*(num_instances//2) + [0]*(num_instances//2)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, 
                                                        test_size=0.2,
                                                        random_state=random_state)

    d_train = lgb.Dataset(X_train, label=y_train)

    grid_params = {
        'learning_rate': (0.1, 0.05, 0.03, 0.01, 0.001),
        'boosting_type': ('goss', 'gbdt', 'dart',),
        'objective': ['binary'],
        'metric':['auc'],
        'num_leaves':(100, 150, 200),
        'max_depth': (50, 100, 200, 300, -1),
        'n_estimators': (50, 100, 200, 300),
        'random_state': [random_state],
    }

    #Grid search nos dados de treino
    lgb_model = lgb.LGBMClassifier()
    clf = GridSearchCV(lgb_model, param_grid=grid_params, scoring="roc_auc")
    print("Começando gridsearch")
    clf.fit(X_train, y_train)

    print(f"grid best_params: {clf.best_params_}")
    print(f"grid best_score: {clf.best_score_}")

    #Treinando um modelo com a melhor configuração
    print(f"Treinando um modelo com os melhores parâmetros sobre todos os dados de treino:")
    lgb_model = lgb.LGBMClassifier(**clf.best_params_)
    lgb_model.fit(X_train, y_train)

    #Avaliando o modelo final
    y_pred = lgb_model.predict(X_test)

    for i in range(X_test.shape[0]):
        if y_pred[i]>= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix: \n{cm}")

    print("Melhor modelo Acc: ", accuracy_score(y_test, y_pred))
    print("Melhor modelo ROCAUC: ", roc_auc_score(y_test, y_pred))

    #Salvando resultados
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    target_results_folder = results_folder / f"run_{timestr}"
    target_results_folder.mkdir(parents=True, exist_ok=True)

    filename = 'trained_model.pkl'
    pickle.dump(lgb_model, open(target_results_folder / filename, 'wb'))

    filename = 'grid_cv.pkl'
    pickle.dump(clf, open(target_results_folder / filename, 'wb'))