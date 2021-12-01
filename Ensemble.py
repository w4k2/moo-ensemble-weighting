
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score as bac
from deap import base, creator, tools, algorithms
from operator import itemgetter
from random import randrange, randint, sample, seed, random
import matplotlib.pyplot as plt


from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination, get_selection


import numpy as np


class PymooProblem(ElementwiseProblem):
    def __init__(self, n_var, y_predicts, y_true):
        self.n_var = n_var
        self.y_predicts = y_predicts
        self.y_true = y_true
        super().__init__(n_var=self.n_var,
                         n_obj=2,
                         n_constr=0,
                         xl=np.full((n_var,), 0.0),
                         xu=np.full((n_var,), 1.0))

    def _evaluate(self, x, out, *args, **kwargs):
        def majority_voting(y_predict):
            acc_y_predict = np.empty((y_predict.shape[1],))
            y_predict = y_predict.astype(int)
            for i in range(y_predict.shape[1]):
                acc_y_predict[i] = np.bincount(y_predict[:, i]).argmax()

            return acc_y_predict

        def weighted_voting(y_predicts, weights):
            weighted_support = (y_predicts * weights[:, np.newaxis, np.newaxis])

            # Acumulate supports
            acumulated_weighted_support = np.sum(weighted_support, axis=0)
            predictions = np.argmax(acumulated_weighted_support, axis=1)
            return predictions
        if np.all(x == 0):
            qual1 = 0
            qual2 = 0
        else:
            y_predict = weighted_voting(self.y_predicts, x)
            qual1 = precision_score(self.y_true.astype("int8"), y_predict.astype("int8"))
            qual2 = recall_score(self.y_true.astype("int8"), y_predict.astype("int8"))


        out["F"] = [-qual1, -qual2]


class MCE(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator_pool=None, no_bags=3,  plot_path="plot", iter_gen=1000, mutation=0.1, population=100,
                 aggregated_metrics=[bac]):
        self._base_estimator_pool = base_estimator_pool
        self._no_bags = no_bags
        self._plot_path = plot_path
        self._iter_gen = iter_gen
        self._mutation = mutation
        self._population = population
        self._aggregated_metrics = aggregated_metrics
        self._no_bags = no_bags
        try:
            self._aggregated_metrics.remove(precision_score)
            self._aggregated_metrics.remove(recall_score)
        except:
            pass

        self._weights = None

        self.ensemble_ = []
        self._ensemble_indices = []
        self.classes_ = None

        self.X_ = None
        self.y_ = None

        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None

        self._y_predict = None
        self._pairwise_diversity_stats = None
        self._ensemble_dict = dict()


    @staticmethod
    def get_group(code, full_list):
        if isinstance(full_list, list):
            return list(np.array(full_list)[np.where(np.array(code) == 1)[0]])
        else:
            return full_list[np.where(np.array(code) == 1)[0]]

    @staticmethod
    def _evaluate_metric(individual, y_predicts, y_true, metric, continous=False):
        if continous:
            y_predict = MCE.weighted_voting(y_predicts,individual)
        else:
            predictions = MCE.get_group(individual, y_predicts)
            y_predict = MCE._majority_voting(predictions)
        qual = metric(y_true.astype("int8"), y_predict.astype("int8"))

        return (qual,)

    def _genetic_optimalisation(self, optimalisation_type='multi', metric=None):
        if optimalisation_type == 'quality_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        elif optimalisation_type == 'precision_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        elif optimalisation_type == 'recall_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))

        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

        IND_SIZE= len(self.ensemble_)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, n=IND_SIZE)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("select", tools.selTournament, tournsize=50)
        toolbox.register("evaluate", MCE._evaluate_metric, y_predicts=self._y_predict, y_true=self._y_valid,
                             metric=metric, continous=True)

        result = algorithms.eaMuCommaLambda(toolbox.population(n=100), toolbox, 100, 100, 0.2, 0.1, self._iter_gen)[0]
        fitnesses = list(map(toolbox.evaluate, result))

        return result, fitnesses

    @staticmethod
    def _majority_voting(y_predict):
        try:
            acc_y_predict = np.empty((y_predict.shape[1],))
            y_predict = y_predict.astype(int)
            for i in range(y_predict.shape[1]):
                acc_y_predict[i] = np.bincount(y_predict[:, i]).argmax()
        except:
            print('UPS')
        return acc_y_predict

    @staticmethod
    def weighted_voting(y_predicts, weights):
        weighted_support = (y_predicts * weights[:, np.newaxis, np.newaxis])

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        predictions = np.argmax(acumulated_weighted_support, axis=1)
        return predictions

    def genetic_algorithm_pymoo(self):
        problem = PymooProblem(len(self.ensemble_), self._y_predict, self._y_valid)
        algorithm = NSGA2(
            pop_size=self._population,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", prob=self._mutation, eta=20),
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", self._iter_gen)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)
        return res

    @staticmethod
    def promethee_function(solutions, criteria_min_max, preference_function, criteria_weights):
        def uni_cal(solutions_col, criteria_min_max, preference_function, criteria_weights):
            uni = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
            uni_weighted = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
            for i in range(np.size(uni, 0)):
                for j in range(np.size(uni, 1)):
                    if i == j:
                        uni[i, j] = 0
                    # Usual preference function
                    elif preference_function == 'u':
                        diff = solutions_col[j] - solutions_col[i]
                        if diff > 0:
                            uni[i, j] = 1
                        else:
                            uni[i, j] = 0
                    uni_weighted[i][j] = criteria_weights * uni[i, j]
            # criteria min (0) or max (1) optimization array
            if criteria_min_max == 0:
                uni_weighted = uni_weighted
            elif criteria_min_max == 1:
                uni_weighted = uni_weighted.T
            return uni_weighted

        weighted_unis = []
        for i in range(solutions.shape[1]):
            weighted_uni = uni_cal(solutions[:, i:i + 1], criteria_min_max[i], preference_function[i], criteria_weights[i])
            weighted_unis.append(weighted_uni)
        agregated_preference = []
        uni_acc = weighted_unis[0]
        uni_cost = weighted_unis[1]
        # Combine two criteria into agregated_preference
        for (item1, item2) in zip(uni_acc, uni_cost):
            agregated_preference.append((item1 + item2)/sum(criteria_weights))
        agregated_preference = np.array(agregated_preference)

        n_solutions = agregated_preference.shape[0] - 1
        # Sum by rows - positive flow
        pos_flows = []
        pos_sum = np.sum(agregated_preference, axis=1)
        for element in pos_sum:
            pos_flows.append(element/n_solutions)
        # Sum by columns - negative flow
        neg_flows = []
        neg_sum = np.sum(agregated_preference, axis=0)
        for element in neg_sum:
            neg_flows.append(element/n_solutions)
        # Calculate net_flows
        net_flows = []
        for i in range(len(pos_flows)):
            net_flows.append(pos_flows[i] - neg_flows[i])
        return net_flows

    def _prune(self):
        marker_list = ['o', 's', 'v', '*', 'P', 'X', 'D']

        results = self.genetic_algorithm_pymoo()
        solutions = results.X
        objectives = results.F
        objectives = [tuple(-obj) for obj in objectives]
        net_flows_precision = self.promethee_function(np.array(objectives), ([1, 1]), (['u', 'u']), ([0.6, 0.4]))
        net_flows_recall = self.promethee_function(np.array(objectives), ([1, 1]), (['u', 'u']), ([0.4, 0.6]))

        def extract_variables(variables):
            extracted = [v[0] for v in variables]
            return extracted

        plt.scatter([f[0] for f in objectives], [f[1] for f in objectives], label='MOO', marker=marker_list[0])

        self._ensemble_dict['precision'] = solutions[objectives.index(max(objectives, key=itemgetter(0)))]
        self._ensemble_dict['recall'] = solutions[objectives.index(max(objectives, key=itemgetter(1)))]
        self._ensemble_dict['balanced'] = solutions[objectives.index(min(objectives, key=lambda i: abs(i[0] - i[1])))]
        self._ensemble_dict['promethee_precision'] = solutions[np.argmax(net_flows_precision, axis=0)]
        self._ensemble_dict['promethee_recall'] = solutions[np.argmax(net_flows_recall, axis=0)]

        pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='precision_single', metric=precision_score)
        self._ensemble_dict['precision_score'] = pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))]
        prec = precision_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict['precision_score']))
        rec = recall_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict['precision_score']))
        plt.scatter(prec, rec,label='precision_single', marker=marker_list[1])

        pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='recall_single', metric=recall_score)
        self._ensemble_dict['recall_score'] = pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))]
        prec = precision_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict['recall_score']))
        rec = recall_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict['recall_score']))
        plt.scatter(prec, rec, label='recall_single', marker=marker_list[2])

        for i_m, m in enumerate(self._aggregated_metrics):
            pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='quality_single', metric=m)
            self._ensemble_dict[m.__name__] = pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))]
            prec = precision_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict[m.__name__]))
            rec = recall_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict[m.__name__]))
            plt.scatter(prec, rec, label=m.__name__+'_single', marker=marker_list[i_m+3])

        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.legend()
        plt.savefig(self._plot_path)
        plt.clf()

        for metric in self._aggregated_metrics:
            pareto_bac = [metric(self._y_valid, self.weighted_voting(self._y_predict, sol)) for sol in solutions]
            plt.scatter([f[0] for f in objectives], pareto_bac, label="MOO", marker=marker_list[0])

            prec = precision_score(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict[metric.__name__]))
            bacc = metric(self._y_valid, self.weighted_voting(self._y_predict, self._ensemble_dict[metric.__name__]))
            plt.scatter(prec, bacc, label=metric.__name__+'_single', marker=marker_list[1])

            plt.xlim([0, 1.1])
            plt.ylim([0, 1.1])
            plt.xlabel("precision")
            plt.ylabel(metric.__name__)
            plt.legend()
            plt.savefig(self._plot_path + '_' + metric.__name__)
            plt.clf()

    def set_ensemble(self, ensemble_type='quality'):
        self._weights = self._ensemble_dict[ensemble_type]

    @staticmethod
    def subsample(X, y, n_sample=None, ratio=1.0):
        if n_sample is None:
            n_sample = round(len(X) * ratio)
        sample_X = np.empty((n_sample, X.shape[1]))
        sample_y = np.empty((n_sample, 1))
        for i in range(n_sample):
            index = randrange(len(X))
            sample_X[i, :] = X[index, :]
            sample_y[i] = y[index]
        return sample_X, sample_y

    @staticmethod
    def stratified_bagging(X, y, ratio=1.0):
        n_sample = round(len(X) * ratio)
        labels = np.unique(y)
        class_n_sample = [round(n_sample*(len(np.where(y == l)[0])/len(y))) for l in labels]

        class_samples = []
        class_samples_label = []

        for class_sample, label in zip(class_n_sample, labels):
            X_samples, y_samples = MCE.subsample(X[np.where(y == label)[0]], y[np.where(y == label)[0]], n_sample=class_sample)
            class_samples.append(X_samples)
            class_samples_label.append(y_samples)

        X_sample = np.concatenate(class_samples)
        y_sample = np.concatenate(class_samples_label)
        return X_sample, y_sample

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        for train_index, test_index in sss.split(X, y):
            self._X_train, self._X_valid = X[train_index], X[test_index]
            self._y_train, self._y_valid = y[train_index], y[test_index]


        for e in self._base_estimator_pool:
            for i in range(self._no_bags):
                #Stratified Bagging
                X_sample, y_sample = MCE.stratified_bagging(self._X_train, self._y_train, 0.4)
                new_e = clone(e)
                new_e.fit(X_sample, y_sample)
                self.ensemble_.append(new_e)



        self._y_predict = np.array([self.fix_proba(member_clf.predict_proba(self._X_valid)) for member_clf in self.ensemble_])
        self._prune()

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        return sum(prediction == y) / len(y)

    def fix_proba(self, proba):
        if len(proba.shape) == 2 and proba.shape[1] == len(self.classes_):
            return proba
        else:
            if proba.shape[1] == 1:
                proba = np.c_[proba, np.ones(proba.shape[0])]
            else:
                for i in range(proba.shape[0]):
                    if proba[i].shape[0] == 1:
                        proba[i] = np.append(proba[i],[0.0])
                proba = np.stack(proba)
            return proba

    def ensemble_support_matrix(self, X):
        """ESM."""
        try:
            result = np.array([self.fix_proba(member_clf.predict_proba(X)) for member_clf in self.ensemble_])
        except:
            print("UPS")

        return result

    def predict_proba(self, X, weights=None):
        """Aposteriori probabilities."""
        if weights is None:
            weights = self._weights
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = (self.ensemble_support_matrix(X) * self._weights[:, np.newaxis, np.newaxis])

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]
