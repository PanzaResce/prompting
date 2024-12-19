from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from datasets import load_dataset, load_from_disk
import logging, pickle

class SVM():
    def __init__(self, dataset, load=False):
        self.clf = None
        self.dataset = dataset

        if load:
            self.load()

    def get_binarized_labels(self, split):
        return [0 if 0 in l else 1 for l in self.dataset[split]["labels"]]

    def train(self):
        # dataset = load_from_disk("./142_dataset/tos.hf/")

        classifier = OneVsRestClassifier(LinearSVC(max_iter=50000, dual=True))
        parameters = {
            'vect__max_features': [10000, 20000, 40000],
            'clf__estimator__C': [0.1, 1, 10],
            'clf__estimator__loss': ('hinge', 'squared_hinge')
        }

        text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=5)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', classifier),
                            ])

        # Fixate Validation Split
        split_index = [-1] * len(self.dataset['train']) + [0] * len(self.dataset['validation'])
        val_split = PredefinedSplit(test_fold=split_index)
        gs_clf = GridSearchCV(text_clf, parameters, cv=val_split, n_jobs=32, verbose=4, refit = False)

        # Pre-process inputs, outputs
        x_train = self.dataset['train']['text']
        x_val = self.dataset['validation']['text']
        
        x_train_val = x_train + x_val
        
        y_train = self.get_binarized_labels(split="train")
        y_val = self.get_binarized_labels(split="validation")
        y_train_val = y_train + y_val

        # Grid search over train + val
        gs_clf = gs_clf.fit(x_train_val, y_train_val)

        # Print best hyper-parameters
        logging.info('Best Parameters:')
        for param_name in sorted(parameters.keys()):
            logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
        # Retrain model with best CV parameters only with train data
        text_clf.set_params(**gs_clf.best_params_)
        gs_clf = text_clf.fit(x_train, y_train)
            
        self.clf = gs_clf

    def predict(self, split="test"):
        y_pred = self.clf.predict(self.dataset[split]["text"])
        y_true = self.get_binarized_labels(split)

        return y_pred, y_true

    def save(self):
        with open('utils/svm.pkl', 'wb') as f:
            pickle.dump(self.clf,f)

    def load(self):
        with open('utils/svm.pkl', 'rb') as f:
            self.clf = pickle.load(f)

if __name__ == '__main__':
    print("TRAINING AND SAVING SVM CLASSIFIER OVER BINARIZED LABELS (FAIR|UNFAIR)")

    dataset = load_from_disk("./142_dataset/tos.hf/")

    clf = SVM(dataset)
    clf.train()
    clf.save()