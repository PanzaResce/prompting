# Prompting on 142ToS Dataset

The `utils/config.py` file holds all the configuration to run the experiments including the prompt templates, the list of models to test (`ENDPOINTS`) and the `Labels` class which handle the operations related to the unfair label categories.

## Necessary steps to run experiments
1) Having an `out/` directory to keep all the experiments' results, they are directly written under this folder
2) Having the dataset in the project root directory (named `142_dataset`)
3) Set the hugginface token in the `prompt_test.py` script
4) Having an environment with all the necessary libraries installed
    - The `Dockerfile` should be enough to handle most of the LLMs

### How to run an experiment (`prompt_test.py`)

For most of the experiments the call to the script is the following:
```bash
python prompt_test.py -all -type prompt_chain -num_shots 8 -quant 8 -device auto
```

For the previous command, the script would write the results, once finished, in the path `out/prompt_chain_8/`.
Specifically, two files are written:
-  ```modelname_score.json```: contains the score of the model both global and per-category
- ```modelname_resp.txt```: contains the response of the model for each input clause, along with the gold label

### Required Arguments  

| Argument  | Type   | Description |
|-----------|--------|-------------|
| `-all`    | Flag   | Run evaluation on all models specified in `ENDPOINTS`. Mutually exclusive with `-m`. |
| `-m`      | String | Specify a single model name for evaluation. Mutually exclusive with `-all`. |
| `-type`   | String | Prompting technique to use. Options: `zero`, `zero_old`, `multi_zero`, `multi_few`, `bare_multi_few`, `prompt_chain`, `svm_few`. |
| `-device` | String | Specifies where to load the model. Options: `cpu`, `cuda`, `auto`. |
| `-quant`  | String | Model quantization setting. Options: `None`, `4`, `8`. |

### Optional Arguments  

| Argument        | Type   | Description |
|----------------|--------|-------------|
| `-subset`      | Int    | Specify a subset of the dataset for evaluation. |
| `-debug`       | Int    | Debug level (default: `0`). |
| `-num_shots`   | Int    | Number of shots for few-shot techniques. Required if `-type` is `multi_few`, `bare_multi_few`, or `prompt_chain`. |
| `-unfair_only` | Flag   | Use a subset containing only unfair clauses. |
| `-resp_file`   | String | Load model responses from `{model}_resp.txt`. |


The script ```pretty_print_report``` can be used to pretty print the result for each experiment.
\
Example usage:
```bash
python pretty_print_report.py -dir multi_few_8_short
```
With the previous command, the metrics for all the experiment concerning the <i>multi_few<i> approach are printed. 