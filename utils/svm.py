from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn import metrics
from datasets import load_dataset, load_from_disk
import logging, pickle, sys
import numpy as np

class SVM():
    def __init__(self, dataset = None, load=False, best_params = {}):
        self.clf = None
        self.dataset = dataset
        self.recall_unfair_threshold = 0.9
        self.f1_unfair_threshold = 0.8
        self.best_params = best_params
        self.scorer = self.make_scorer()

        if load:
            self.load()

    def get_binarized_labels(self, dataset):
        return [0 if 0 in l else 1 for l in dataset["labels"]]

    def recall_strategy(self, gs_results):
        avg_recall = (gs_results["mean_test_recall_fair"] + gs_results["mean_test_recall_unfair"])/2
        best_index = avg_recall.argmax()
        print(best_index)
        if gs_results["mean_test_recall_unfair"][best_index] <= self.recall_unfair_threshold:
            print(f"Best model has recall under the threshold: {gs_results['mean_test_recall_unfair'][best_index]}|{self.recall_unfair_threshold}")

        print(f"OTHER MODELS WITH UNFAIR RECALL OVER THE THRESHOLD {self.recall_unfair_threshold}")
        for i, unfair_rec in enumerate(gs_results["mean_test_recall_unfair"]):
            if i != best_index and unfair_rec > self.recall_unfair_threshold:
                print(f"Recall Fair:\t{gs_results['mean_test_recall_fair'][i]:.2f}")
                print(f"Recall Unfair:\t{gs_results['mean_test_recall_unfair'][i]:.2f}")
                print(f"Params:\t{gs_results['params'][i]}")

        print("BEST MODEL PARAMETERS")
        print(f"Recall Fair:\t{gs_results['mean_test_recall_fair'][best_index]:.2f}")
        print(f"Recall Unfair:\t{gs_results['mean_test_recall_unfair'][best_index]:.2f}")
        print(f"Params:\t{gs_results['params'][best_index]}")
        return best_index

    def f1_strategy(self, gs_results):
        best_index = gs_results["mean_test_f1_unfair"].argmax()
        if gs_results["mean_test_f1_unfair"][best_index] <= self.recall_unfair_threshold:
            print(f"Best model has f1 under the threshold: {gs_results['mean_test_f1_unfair'][best_index]}|{self.f1_unfair_threshold}")
        print(f"OTHER MODELS WITH UNFAIR F1 OVER THE THRESHOLD {self.recall_unfair_threshold}")
        for i, f1_rec in enumerate(gs_results["mean_test_f1_unfair"]):
            if i != best_index and f1_rec > self.f1_unfair_threshold:
                print(f"F1 Fair:\t{gs_results['mean_test_f1_fair'][i]:.2f}")
                print(f"F1 Unfair:\t{gs_results['mean_test_f1_unfair'][i]:.2f}")
                print(f"Params:\t{gs_results['params'][i]}")
        return best_index

    def refit_strategy(self, gs_results, strategy_type):
        print(f"Strategy: {strategy_type}")
        if strategy_type == "recall_unfair":
            best_index = self.recall_strategy(gs_results)
        elif strategy_type == "f1_unfair":
            best_index = self.f1_strategy(gs_results)
        return best_index

    def make_scorer(self):
        recall_fair_scorer = metrics.make_scorer(metrics.recall_score, average=None, labels=[0])
        recall_unfair_scorer = metrics.make_scorer(metrics.recall_score, average=None, labels=[1])
        f1_fair_scorer = metrics.make_scorer(metrics.f1_score, average=None, labels=[0])
        f1_unfair_scorer = metrics.make_scorer(metrics.f1_score, average=None, labels=[1])

        scoring = {
            'recall_fair': recall_fair_scorer,
            'recall_unfair': recall_unfair_scorer,
            'f1_fair': f1_fair_scorer,
            'f1_unfair': f1_unfair_scorer
        }
        return scoring

    def train(self, input_parameters={}):
        classifier = SVC(max_iter=50000, class_weight="balanced")

        text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=5)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', classifier),
                            ])
        
        x_train = self.dataset['train']['text']
        x_val = self.dataset['validation']['text']
        y_train = self.get_binarized_labels(self.dataset['train'])
        y_val = self.get_binarized_labels(self.dataset['validation'])

        if not input_parameters:
            parameters = [
                {"clf__kernel": ["rbf"], "clf__gamma": np.logspace(-4, 3, 10).tolist(), "clf__C": np.logspace(-2, 4, 10).tolist()},
                {"clf__kernel": ["linear"], "clf__C": np.logspace(-2, 4, 10).tolist()},
            ]

            # parameters = [
            #     {"clf__kernel": ["linear"]},
            #     {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1]},
            # ]

            # Fixate Validation Split
            split_index = [-1] * len(self.dataset['train']) + [0] * len(self.dataset['validation'])
            val_split = PredefinedSplit(test_fold=split_index)
            gs_clf = GridSearchCV(text_clf, parameters, cv=val_split, n_jobs=32, verbose=4, refit=lambda res: self.refit_strategy(res, 'f1_unfair'), scoring=self.scorer)
            
            x_train_val = x_train + x_val
            y_train_val = y_train + y_val

            # Grid search over train + val
            gs_clf = gs_clf.fit(x_train_val, y_train_val)

            # Print best hyper-parameters
            print('Best Parameters:')
            for param_name, value in gs_clf.best_params_.items():
                print("%s: %r" % (param_name, value))
            best_params = gs_clf.best_params_
        else:
            best_params = input_parameters

        print(f"Training final model with best parameters only with train data\n{best_params}")
        text_clf.set_params(**best_params)
        gs_clf = text_clf.fit(x_train, y_train)
            
        self.clf = gs_clf

    def predict(self, dataset):
        y_pred = self.clf.predict(dataset["text"])
        return y_pred

    def evaluate(self, dataset, report=False):
        y_pred = self.predict(dataset)
        y_true = self.get_binarized_labels(dataset)

        if report:
            return metrics.classification_report(y_true, y_pred, zero_division=0, target_names=["fair", "unfair"])   
        else:
            print(f"Micro F1: {metrics.f1_score(y_true, y_pred, average='micro')*100:.2f}")
            print(f"Macro F1: {metrics.f1_score(y_true, y_pred, average='macro')*100:.2f}")
            return y_pred, y_true

    def save(self):
        with open('utils/svm.pkl', 'wb') as f:
            pickle.dump(self.clf,f)

    def load(self):
        with open('utils/svm.pkl', 'rb') as f:
            self.clf = pickle.load(f)

def evaluate_svm(dataset):
    ds_test = dataset["test"]

    clf = SVM(load=True)
    report = clf.evaluate(ds_test, report=True)
    print(report)

if __name__ == '__main__':
    # rec balanced
    # default_best_params = {
    #     "clf__kernel": "rbf",
    #     "clf__C": 1.0,
    #     "clf__gamma": 0.003593813663804626
    # }

    # rec 1
    default_best_params = {
        "clf__kernel": "rbf",
        "clf__C": 21.54434690031882,
        "clf__gamma": 0.0001
    }

    # f1 - 0.71
    # default_best_params = {
    #     'clf__C': 1.0, 
    #     'clf__gamma': 0.774263682681127, 
    #     'clf__kernel': 'rbf'
    # }
    

    dataset = load_from_disk("./142_dataset/tos.hf/")
    clf = SVM(dataset)
    if len(sys.argv) > 1: 
        if sys.argv[1] == "-best":
            print(f"Training with default parameters: {default_best_params}")
            clf.train(default_best_params)
            clf.save()
        elif sys.argv[1] == "-eval":
            print("Evaluating model svm.pkl located in utils/")
            evaluate_svm(dataset)
    else:
        clf.train()
        clf.save()