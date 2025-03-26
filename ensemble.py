import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utils.config import LABELS
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report
from pretty_print_report import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretty print report.")
    
    parser.add_argument(
        "-dir",
        type=str,
        required=True,
        metavar="dir",
        help="Directory containing the resp.txt files"
    )

    parser.add_argument(
        "-graphs",
        type=str,
        required=False,
        help="Produce graphs with title"
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-s",
        action="store_true",
        help="Ensemble only small models"
    )
    group.add_argument(
        "-l",
        action="store_true",
        help="Ensemble only large model"
    )
    group.add_argument(
        "-b",
        action="store_true",
        help="Ensemble with the best models"
    )
    
    return parser.parse_args()

def load_data(file_path):
    """Load and parse the input file."""
    true_label_matrix = []
    predicted_label_matrix = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                _, true_label, predicted_labels = parts[-3:]
                true_label_set = set(true_label.strip('[]').split(','))
                predicted_label_set = set(predicted_labels.strip('[]').split(','))
                
                # Convert labels to binary indicator vectors
                true_vector = [1 if label in true_label_set else 0 for label in LABELS.labels]
                predicted_vector = [1 if label in predicted_label_set else 0 for label in LABELS.labels]

                true_label_matrix.append(true_vector)
                predicted_label_matrix.append(predicted_vector)
                
    return np.array(true_label_matrix), np.array(predicted_label_matrix)

def hard_voting(classifiers, type, threshold):
    # print(classifiers)
    # for k,v in classifiers.items():
    #     print(v[1])
    # l = list()
    for k,v in classifiers.items():
        if k == "codestral":
            s=v[1].shape
    per_model_predictions = np.array([v[1] for k,v in classifiers.items()])
    true_labels = list(classifiers.values())[0][0]

    if type == "softmax":
        exp = np.exp(per_model_predictions.sum(axis=0))
        predictions =  exp / exp.sum(axis=-1)[:, np.newaxis]
        predictions[predictions < threshold] = 0
        predictions[predictions >= threshold] = 1
    elif type == "hard": 
        predictions = per_model_predictions.sum(axis=0)
        predictions[predictions < threshold] = 0
        predictions[predictions >= threshold] = 1
    return classification_report(true_labels, predictions, zero_division=0, target_names=LABELS.labels_to_id().keys(), output_dict=True)

def weighted_voting(classifiers, threshold, macro_f1s):
    macro_f1s = list(macro_f1s.values())
    weights = np.array([f1/sum(macro_f1s) for f1 in macro_f1s])

    per_model_predictions = np.array([v[1] for k,v in classifiers.items()])
    true_labels = list(classifiers.values())[0][0]

    weighted_predictions = per_model_predictions * weights[:, np.newaxis, np.newaxis]
    
    summed_predictions = weighted_predictions.sum(axis=0)
    summed_predictions[summed_predictions < threshold] = 0
    summed_predictions[summed_predictions >= threshold] = 1
    return classification_report(true_labels, summed_predictions, zero_division=0, target_names=LABELS.labels_to_id().keys(), output_dict=True)

def grid_search(classifiers):
    grid_search = {}
    for i in np.linspace(0.1,1,100):
        out = hard_voting(classifiers, "softmax", i)
        grid_search[str(i)] = out["macro avg"]["f1-score"]
    grid_search = dict(sorted(grid_search.items(), key=lambda item: item[1], reverse=True))
    best_threshold = list(grid_search.keys())[0]
    return best_threshold

def get_thresholds(classifiers):
    best_threshold = float(grid_search(classifiers))
    hard_threshold = round(len(classifiers)/2)+1 if len(classifiers)%2 == 0 else round(len(classifiers)/2)
    weight_threshold = 0.5
    return best_threshold, hard_threshold, weight_threshold

def n_ensemble_graph(classifiers, model_f1, title):
    # n_ensembles = 4
    soft_votes = list()
    hard_votes = list()
    weight_votes = list()

    for n_ensembles in range(2, len(classifiers.keys())):
        print(f"N.ensembles: {n_ensembles}")
        best_classifiers = {}
        best_f1 = {}

        sorted_f1 = [v for k, v in sorted(model_f1.items(), key=lambda item: item[1], reverse=True)]
        min_f1 = sorted_f1[n_ensembles-1]

        for k in list(classifiers.keys()):
            if model_f1[k] >= min_f1:
                best_classifiers[k] = classifiers[k]
                best_f1[k] = model_f1[k]
        
        # print(best_classifiers, best_f1)

        soft_threshold, hard_threshold, weight_threshold = get_thresholds(best_classifiers)

        v_soft = hard_voting(best_classifiers, "softmax", soft_threshold)
        v_hard = hard_voting(best_classifiers, "hard", hard_threshold)
        v_weight = weighted_voting(best_classifiers, weight_threshold, best_f1)

        soft_votes.append(v_soft["macro avg"]["f1-score"])
        hard_votes.append(v_hard["macro avg"]["f1-score"])
        weight_votes.append(v_weight["macro avg"]["f1-score"])


    figure(figsize=(12, 10))
    plt.plot(range(2, len(classifiers.keys())), soft_votes, label="softmax")
    plt.plot(range(2, len(classifiers.keys())), hard_votes, label="hard")
    plt.plot(range(2, len(classifiers.keys())), weight_votes, label="weighted")

    plt.axhline(sorted_f1[0], label="best model", color="r", linestyle="--")

    plt.legend(loc='best')
    plt.xlabel("N.Ensembles")
    plt.ylabel("Macro F1")
    plt.title(title)
    # plt.show()
    plt.savefig(f'ensembles/{title.replace("-","_").lower()}.png', bbox_inches='tight')

def main():
    args = parse_arguments()
    
    directory_path = f"out/{args.dir}"

    classifiers = {}
    model_f1 = {}
    for filename in os.listdir(directory_path):
        model_name = filename.split("_")[0]
        ext = filename.split(".")[-1]
        if ext == "txt":
            # read resp.txt file
            file_path = os.path.join(directory_path, filename)
            true_matrix, pred_matrix = load_data(file_path)
            classifiers[model_name] = [true_matrix, pred_matrix]
        elif ext == "json":
            # retrieve macro-F1
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                report = json.load(file)
            model_f1[model_name] = float(report[model_name]["test"]["report"]["macro avg"]["f1-score"])


    if args.s:
        del classifiers["codestral"], model_f1["codestral"]
        del classifiers["qwen32"], model_f1["qwen32"]
        print(f"Using only smaller models: {list(classifiers.keys())}")
    elif args.l:
        del classifiers["lawllm"], model_f1["lawllm"]
        del classifiers["mistral"], model_f1["mistral"]
        del classifiers["nemo"], model_f1["nemo"]
        del classifiers["phi3"], model_f1["phi3"]
        del classifiers["llama8"], model_f1["llama8"]
        del classifiers["gemma2"], model_f1["gemma2"]
        del classifiers["gemma9"], model_f1["gemma9"]
        print(f"Using only larger models: {list(classifiers.keys())}")
    elif args.b:
        if args.graphs == None:
            # del classifiers["qwen32"], model_f1["qwen32"]
            n_ensembles = 4
            sorted_f1 = [v for k, v in sorted(model_f1.items(), key=lambda item: item[1], reverse=True)]
            min_f1 = sorted_f1[n_ensembles-1]
            for k in list(classifiers.keys()):
                if model_f1[k] < min_f1:
                    del classifiers[k], model_f1[k]
            print(f"Ensembling on the {n_ensembles} best models: {list(classifiers.keys())}")
    else:
        print(f"Ensembling with all the models: {list(classifiers.keys())}")
    
    # print(model_f1)
    # best_threshold = float(grid_search(classifiers))
    # hard_threshold = round(len(classifiers)/2)+1 if len(classifiers)%2 == 0 else round(len(classifiers)/2)
    # weight_threshold = 0.5
    if not args.graphs == None:
        n_ensemble_graph(classifiers, model_f1, args.graphs)
    else:
        best_threshold, hard_threshold, weight_threshold = get_thresholds(classifiers)

        v_soft = hard_voting(classifiers, "softmax", best_threshold)
        v_hard = hard_voting(classifiers, "hard", hard_threshold)
        v_weight = weighted_voting(classifiers, weight_threshold, model_f1)

        print(f"---Softmax [{best_threshold:.2f}]---")
        print_model_f1(v_soft)
        print()

        print(f"---Hard [{hard_threshold}]---")
        print_model_f1(v_hard)
        print()

        print(f"---Weighted [{weight_threshold}]---")
        print_model_f1(v_weight)

if __name__ == "__main__":
    main()