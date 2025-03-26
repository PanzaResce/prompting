import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from utils.config import LABELS
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

# Define the labels

# Load the data from a file
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretty print report.")
    
    parser.add_argument(
        "-dir",
        type=str,
        required=True,
        metavar="dir",
        help="Directory containing the resp.txt files"
    )
    
    return parser.parse_args()


def normalize_along_diag(m):
    matrix_copy = m.astype(float)

    main_d = np.diag(m) / np.trace(m)
    anti_d = np.fliplr(m).diagonal()
    anti_d = anti_d / sum(anti_d)

    np.fill_diagonal(matrix_copy, main_d)
    np.fill_diagonal(np.fliplr(matrix_copy), anti_d)
    return matrix_copy

def normalize_along_rows(m):
    return m.astype(float) / m.sum(axis=1, keepdims=True)

def main():
    args = parse_arguments()

    directory_path = "out/"+args.dir
    print(f"Drawing conf matrix for {directory_path}")

    for filename in os.listdir(directory_path):
        ext = filename.split(".")[-1]
        if ext != "txt":
            continue
        
        model_name = filename.split("_")[0]
        fig, axs = plt.subplots(4,3)
        # fig.suptitle(model_name+" Categories Confusion Matrix", fontsize=16)
        fig.suptitle(f"Model:    {model_name}", fontsize=16, y=0.93)
        # fig.suptitle(f"Model:    {model_name}\nMethod:    {args.dir}", fontsize=16)
        axs[3,1].axis('off')
        axs[3,2].axis('off')
        fig.set_size_inches(12, 14)

        file_path = os.path.join(directory_path, filename)
        true_matrix, pred_matrix = load_data(file_path)
        mcm = multilabel_confusion_matrix(true_matrix, pred_matrix)
        for idx, matrix in enumerate(mcm):
            row, col = idx//3, idx%3
            # print(f"Confusion Matrix for label '{LABELS.labels[idx]}':\n{matrix}")
            # n_matrix = normalize_along_diag(matrix)
            n_matrix = normalize_along_rows(matrix)
            disp = ConfusionMatrixDisplay(confusion_matrix=n_matrix, display_labels=[0, 1])
            disp.plot(ax=axs[row, col], cmap='BuPu')

            for i, labels in enumerate(disp.text_.ravel()):
                if i == 1 or i == 2:
                    labels.set_text(labels._text + f"\n({matrix.ravel()[i]}/{sum(np.fliplr(matrix).diagonal())})")
                labels.set_fontsize(15)

            # axs[row, col].plot()
            disp.ax_.set_title(LABELS.labels[idx], fontsize=16)
            disp.im_.colorbar.remove()
            if idx == len(mcm)-1:
                disp.ax_.set_xlabel('Predicted', fontsize=14)
                disp.ax_.set_ylabel('True', fontsize=14)
            else:
                disp.ax_.set_xlabel('')
                disp.ax_.set_ylabel('')
            # if i!=0:
            #     disp.ax_.set_ylabel('')
            # axs[row, col].set_title()

        plt.subplots_adjust(wspace=0, hspace=0.3)
        # fig.colorbar(disp.im_, ax=axs)
        plt.savefig(os.path.join(directory_path, model_name+"_matrix.png"))

if __name__ == "__main__":
    main()
