from huggingface_hub import login

import argparse, json
from utils.utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run evaluation on specified models.")
    
    # Mutually exclusive group for either -all or -m [model_name]
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-all",
        action="store_true",
        help="Run evaluation on all models specified in ENDPOINTS"
    )
    group.add_argument(
        "-m",
        type=str,
        metavar="model_name",
        help="Specify a single model name to run the evaluation"
    )

    parser.add_argument(
        "-subset",
        type=int,
        required=False,
        metavar="dataset_subset",
        help="Specify to use a subset of the dataset"
    )

    parser.add_argument(
        "-type",
        type=str,
        required=True,
        metavar="prompting_technique",
        choices=["zero", "zero_old", "multi_zero", "multi_few", "multi_few_half"],
        help="Specify the prompting technique to use"
    )

    parser.add_argument(
        "-quant",
        type=str,
        required=True,
        metavar="quantization",
        choices=["None", "4", "8"],
        help="Specify the quantization bits"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Token login")
    login("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")

    print("Loading dataset")
    ds_test = load_dataset("./142_dataset/tos.hf/", split="test")

    if args.subset != "":
        print(f"Using a subset of the dataset: {args.subset}")
        ds_test = ds_test[:args.subset]
    # ds_val = load_dataset("../dataset_refactoring/142_dataset/tos.hf/", split="validation")

    if args.all:
        print("Running evaluation on all models.")
        models_score = evaluate_models(ENDPOINTS, ds_test, None, args.type, args.quant)

        print(models_score)
        write_res_to_file("models", models_score, args.type)


    elif args.m:
        print(f"Running evaluation on model: {args.m}")
        endpoint = {f"{extract_model_name(args.m)}": f"{args.m}"}
        model_score = evaluate_models(endpoint, ds_test, None, args.type, args.quant)

        print(model_score)
        write_res_to_file(args.m, model_score, args.type)
