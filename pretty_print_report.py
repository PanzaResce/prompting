import argparse, os, json
from tabulate import tabulate

def print_classification_report_tabulate(scikit_classification_report):
    headers = ["Label", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for label, metrics in scikit_classification_report.items():
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1_score = metrics.get("f1-score", 0.0)
        support = metrics.get("support", 0.0)
        rows.append([label, f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{support:.0f}"])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def print_model_f1(scikit_classification_report):
    print(f"Macro F1: {scikit_classification_report["macro avg"]["f1-score"]*100:.1f}")
    print(f"Micro F1: {scikit_classification_report["micro avg"]["f1-score"]*100:.1f}")
    print(f"Macro Precision: {scikit_classification_report["macro avg"]["precision"]*100:.1f}")
    print(f"Micro Precision: {scikit_classification_report["micro avg"]["precision"]*100:.1f}")

def pretty_print_classification_report(scikit_classification_report, only_metrics):
    if only_metrics:
        print_model_f1(scikit_classification_report)
    else:
        print_classification_report_tabulate(scikit_classification_report)
    


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretty print report.")
    
    parser.add_argument(
        "-dir",
        type=str,
        required=True,
        metavar="prompting_technique",
        help="Specify the prompting technique to use"
    )

    parser.add_argument(
        "-f1",
        action="store_true",
        help="Output only the macro/micro F1"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    directory_path = f"out/{args.dir}"
    for filename in os.listdir(directory_path):
        ext = filename.split(".")[-1]
        if ext != "json":
            continue
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            report = json.load(file)
            model_name = list(report.keys())[0]
            print(f"------------------------Report for model {model_name}------------------------")
            pretty_print_classification_report(report[model_name]["test"]["report"], args.f1)
