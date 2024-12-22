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
        self.best_params = best_params
        self.scorer = self.make_scorer()

        if load:
            self.load()

    def get_binarized_labels(self, dataset):
        return [0 if 0 in l else 1 for l in dataset["labels"]]

    def refit_strategy(self, gs_results):
        # filtered_recall_fair = gs_results["mean_test_recall_fair"][gs_results["mean_test_recall_unfair"] > self.recall_unfair_threshold]
        # filtered_recall_unfair = gs_results["mean_test_recall_unfair"][gs_results["mean_test_recall_unfair"] > self.recall_unfair_threshold]
        # print(f"{filtered_recall_fair}")
        # print(f"{filtered_recall_unfair}")
        
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

    def make_scorer(self):
        recall_fair_scorer = metrics.make_scorer(metrics.recall_score, average=None, labels=[0])
        recall_unfair_scorer = metrics.make_scorer(metrics.recall_score, average=None, labels=[1])

        scoring = {
            'recall_fair': recall_fair_scorer,
            'recall_unfair': recall_unfair_scorer
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
            gs_clf = GridSearchCV(text_clf, parameters, cv=val_split, n_jobs=32, verbose=4, refit=self.refit_strategy, scoring=self.scorer)
            
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

if __name__ == '__main__':
    print("TRAINING AND SAVING SVM CLASSIFIER OVER BINARIZED LABELS (FAIR|UNFAIR)")
    default_best_params = {
        "clf__kernel": "rbf",
        "clf__C": 1.0,
        "clf__gamma": 0.003593813663804626
    }

    # default_best_params = {
    #     "clf__kernel": "rbf",
    #     "clf__C": 21.54434690031882,
    #     "clf__gamma": 0.0001
    # }

    dataset = load_from_disk("./142_dataset/tos.hf/")
    clf = SVM(dataset)
    if sys.argv[1] == "-best":
        print(f"Training with default parameters: {default_best_params}")
        clf.train(default_best_params)
    else:
        clf.train()
    clf.save()