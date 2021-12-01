from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score as gmean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from collections import Counter

import pandas as pd
import numpy as np
import os
import argparse
from DatasetsCollection import load
from Ensemble import MCE


file_list = ["abalone9-18", "glass4", "glass5", "yeast4", "yeast5", "yeast6", "flare-F", "ecoli1", "ecoli2", "ecoli3",
             "glass0", "glass1", "haberman", "page-blocks0", "pima", "vehicle1", "vehicle3", "yeast1", "yeast3"]

data_set = []

metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, gmean]


def experiment(data, classifiers, plot_path="plot", small_data=True, **kwargs):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    pruning_strategy_list = ['precision','recall','balanced', 'promethee_precision', 'promethee_recall'] + [m.__name__ for m in metrics]
    metrics_array = np.empty((len(pruning_strategy_list), 10, len(metrics)))

    if not small_data:
        i = 0
        for train_index, test_index in sss.split(data[0], data[1]):
            for j in range(2):
                mce = MCE(base_estimator_pool=classifiers, plot_path=plot_path+str(i), aggregated_metrics=metrics.copy(), **kwargs)
                mce.fit(data[0][train_index], data[1][train_index])

                for inp, p in enumerate(pruning_strategy_list):
                    mce.set_ensemble(p)
                    y_predict = mce.predict(data[0][test_index])
                    for ind, it in enumerate(metrics):
                        metrics_array[inp, i, ind] = it(data[1][test_index], y_predict)
                i +=1
                train_index, test_index = test_index, train_index
    else:
        for i, fold in enumerate(data):
            mce = MCE(base_estimator_pool=classifiers,  plot_path=plot_path+"_"+str(i+1), aggregated_metrics=metrics.copy(), **kwargs)
            mce.fit(fold[0][0], fold[0][1])
            for inp, p in enumerate(pruning_strategy_list):
                mce.set_ensemble(p)
                y_predict = mce.predict(fold[1][0])
                for ind, it in enumerate(metrics):
                    metrics_array[inp, i, ind] = it(fold[1][1], y_predict)

    return np.mean(metrics_array, axis=1)


def prepare_dataset(whole_ds, class_label):
    try:
        whole_ds.drop(columns='Unnamed: 0', inplace=True)
    except:
        pass
    y = whole_ds[class_label].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    c = Counter(y)
    if c[1] > c[0]:
        y = abs(y-1)
    cols = list(whole_ds.columns)
    cols.remove(class_label)
    X = whole_ds[cols].values
    return X, y


def get_datasets(data_file=None):
    if data_file is None:
        for file in file_list:
            data_set.append(load(file))
    else:
        df = pd.read_csv(data_file)
        data_set.append(df)


def prepare_classifiers():
    ada = AdaBoostClassifier(random_state=42)
    r_forest = RandomForestClassifier(random_state=4)
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    ann = MLPClassifier(random_state=666)
    tree = DecisionTreeClassifier(random_state=7)
    return [ada, r_forest, nb, knn, ann, tree]


def conduct_experiments(result_directory="wyniki", data_file=None, bags=3, mutation=0.1, population=100):
    classifiers = prepare_classifiers()
    get_datasets(data_file)
    pruning_strategy = ['precision','recall','balanced', 'promethee_precision', 'promethee_recall'] + [m.__name__ for m in metrics]

    if data_file is None:
        small_data = True
    else:
        small_data = False
        file_list = [os.path.basename(data_file).split('.')[0]]

    results = [pd.DataFrame(index=file_list, columns=pruning_strategy) for m in metrics]


    try:
        os.mkdir(result_directory)
    except:
        pass
    for file_name, data in zip(file_list, data_set):
        file_directory = os.path.join(result_directory,file_name.split('.')[0])
        try:
            os.mkdir(file_directory)
        except:
            pass
        if data_file is not None:
            X, y = prepare_dataset(data, "Class")
            data = (X, y)

        r = experiment(data, classifiers, no_bags=bags, plot_path=os.path.join(file_directory, file_name.split('.')[0]),
                       iter_gen=500, mutation=mutation, population=population, small_data=small_data)
        for i_m, m in enumerate(metrics):
            for i, s in enumerate(pruning_strategy):
                results[i_m].at[file_name, s] = r[i, i_m]

        for i_m, m in enumerate(metrics):
            results[i_m].to_csv(os.path.join(file_directory, file_name.split('.')[0] + "_results_"+m.__name__+".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--bags", type=int, default=3)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--mutation", type=float, default=0.1)
    args = parser.parse_args()

    conduct_experiments(args.result_directory, args.data_file, args.bags, args.mutation, args.population)



        













































