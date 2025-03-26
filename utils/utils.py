import transformers, json, torch, re, os
from datasets import load_from_disk
from sklearn.metrics import classification_report
from .config import *
from .prompt_manager import *
from .prompt_consumer import * 
from .svm import SVM

def extract_model_name(card):
    return card.split("/")[-1]

def get_bnb_config(num_bit: str):
    admissable_quant = ("8", "4", "None")
    if num_bit not in admissable_quant:
        print(f"Quantization type must be one of {admissable_quant}, {type} not valid")
    
    if num_bit == "8":
        return transformers.BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif num_bit == "4":
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        return

def load_dataset(path, split=None):
    ds = load_from_disk(path)
    if split:
        return ds[split]
    return ds

def pick_only_unfair_clause(ds_test, num_elements=None):    
    new_ds_test = {"text": [], "labels": []}
    for clause, label in zip(ds_test["text"], ds_test["labels"]):
        if 0 not in label:
            new_ds_test["text"].append(clause)
            new_ds_test["labels"].append(label)

    if num_elements is None:
        num_elements = len(new_ds_test["text"])

    return {"text": new_ds_test["text"][:num_elements], "labels": new_ds_test["labels"][:num_elements]}

def load_model_and_tokenizer(model_card, bnb_config, device_map):
    torch.cuda.empty_cache()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.generation_config.do_sample=False
    model.generation_config.top_p=1.0
    model.generation_config.temperature=None
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_card)
    return model, tokenizer


def compute_f1_score(responses, labels, label_to_id, debug=0):
    y_true, y_pred = [], []
    for r, label in zip(responses, labels):
        if not isinstance(r, list):
            resp_tags = re.findall(r"(?<=<)[^>]+(?=>)", r)
        else:
            resp_tags = r

        if len(resp_tags) == 0:
            resp_tags = r.split(",")

        true_sample = [0] * len(label_to_id)
        for t in label:
            # Labels in the original dataset range from 0 to 9 where 0 is fair
            true_sample[t] = 1

        pred_sample = [0] * len(label_to_id)
        for t in resp_tags:
            if t in label_to_id.keys():
                pred_sample[label_to_id[t]] = 1
            else:
                if debug > 2:
                    print(f"'{t}' is not a valid tag ({list(label_to_id.keys())})")

        if debug > 0:
            print(f"Extracted tags: {resp_tags}")

        if debug > 1:
            print(f"Sample encoding: {pred_sample}")
            print(f"True encoding: {true_sample}")

        if debug > 2:
            print(f"Original response: {r}")

        y_true.append(true_sample)
        y_pred.append(pred_sample)

    report = classification_report(y_true, y_pred, zero_division=0, target_names=label_to_id.keys(), output_dict=True)
    return {"report": report}


def write_predictions_to_file(responses, dataset, output_dir, model_name):
    os.makedirs(f"out/{output_dir}/", exist_ok=True)
    with open(f'out/{output_dir}/{model_name}_resp.txt', 'w') as file: 
        for clause, label, resp in zip(dataset["text"], dataset["labels"], responses):
            labels = ",".join([LABELS.labels[l] for l in label])
            resps = ",".join(resp)
            # clause, true labels, predicted labels
            to_write = clause + f"\t[{labels}]" + f"\t[{resps}]\n"
            file.write(to_write)

def write_res_to_file(model_name, model_score, output_dir):
    os.makedirs(f"out/{output_dir}/", exist_ok=True)
    with open(f'out/{output_dir}/{model_name}_score.json', 'w') as file: 
        file.write(json.dumps(model_score))


def read_examples_from_json(exclude_fair):
    with open("utils/examples.json", "r") as f:
        per_category_examples = json.load(f)
    
    if exclude_fair:
        del per_category_examples["fair"]
        return per_category_examples
    else:
        return per_category_examples


def get_prompt_manager(prompt_type, label_to_id, num_shots):
    if prompt_type == "zero":
        prompt_manager = StandardPrompt(ZERO_PROMPT, max_new_tokens=30)
    # elif prompt_type == "multi_few_half":
    #     prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW, label_to_id, num_shots=2, only_unfair_examples=False)
    elif prompt_type == "multi_few":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW, label_to_id, per_category_examples, num_shots, only_unfair_examples=True, max_new_tokens=2)
        prompt_manager.set_response_type(positive="yes", negative="no")
    elif prompt_type == "bare_multi_few":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW_NO_TEMPL, label_to_id, per_category_examples, num_shots, only_unfair_examples=True, max_new_tokens=2)
        prompt_manager.set_response_type(positive="yes", negative="no")
        # prompt_manager.set_response_type(positive="y", negative="n")
    elif prompt_type == "multi_zero":
        prompt_manager = ZeroMultiPrompt(MULTI_PROMPT, label_to_id, max_new_tokens=2)
        prompt_manager.set_response_type(positive="yes", negative="no")
    elif prompt_type == "prompt_chain":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        prompt_manager_0 = ZeroMultiPrompt(PROMPT_CHAIN_0, label_to_id, max_new_tokens=2)
        prompt_manager_0.set_response_type(positive="yes", negative="no")

        prompt_manager_1 = FewMultiPrompt(PROMPT_CHAIN_1, label_to_id, per_category_examples, num_shots, only_unfair_examples=True, max_new_tokens=2)
        prompt_manager_1.set_response_type(positive="yes", negative="no")

        prompt_manager = [prompt_manager_0, prompt_manager_1]
    else:
        return None

    return prompt_manager

def factory_instantiate_prompt_consumer(prompt_type, model, tokenizer, num_shots, label_to_id, resp_file):
    if resp_file == "":
        model_has_chat_template = not tokenizer.chat_template is None
        if not model_has_chat_template:
            print("Initializing default template")
            tokenizer.chat_template = PromptConsumer.default_template
        print(f"Model has chat template: {model_has_chat_template}")

    prompt_manager = get_prompt_manager(prompt_type, label_to_id, num_shots)

    if prompt_type == "prompt_chain":
        prompt_consumer = PromptChainConsumer(prompt_manager, model, tokenizer)
    elif prompt_type == "svm_few":
        prompt_manager = get_prompt_manager("multi_few", label_to_id, num_shots)
        svm = SVM(load=True)
        prompt_consumer = SVMPromptConsumer(prompt_manager, model, tokenizer, svm, resp_file)
    elif prompt_type == "multi_few_no_templ":
        prompt_consumer = BareModelPromptConsumer(prompt_manager, model, tokenizer)
    else:
        prompt_consumer = PipelinePromptConsumer(prompt_manager, model, tokenizer)
    
    print(f"{type(prompt_consumer).__name__} is being used")
    return prompt_consumer


def evaluate_models(endpoints, ds_test, ds_val, prompt_type, quant, num_shots, device_map, write_to_file, resp_file, debug=0):
    label_to_id = LABELS.labels_to_id()

    bnb_config = get_bnb_config(quant)
    models_score = {}

    # prompt_manager = get_prompt_manager(prompt_type, num_shots)

    for model_name, model_card in endpoints.items():
        print(f"------------ Evaluating model {model_name} on {prompt_type} with {quant} bit quantization ------------")

        if resp_file == None:
            model, tokenizer = load_model_and_tokenizer(model_card, bnb_config, device_map)
        else:
            model, tokenizer = None, None
        prompt_consumer = factory_instantiate_prompt_consumer(prompt_type, model, tokenizer, num_shots, label_to_id, resp_file)
        test_responses = prompt_consumer.generate_responses(ds_test, debug)
        test_report = compute_f1_score(test_responses, ds_test["labels"], label_to_id, debug)

        validation_report = {"report": {}}

        models_score[model_name] = {
            "test": test_report,
            "validation": validation_report
        }

        output_dir = prompt_type if num_shots is None else prompt_type+"_"+str(num_shots)
        if write_to_file:
            write_res_to_file(model_name, {model_name: {"test": test_report,"validation": validation_report}}, output_dir)
        write_predictions_to_file(test_responses, ds_test, output_dir, model_name)

        if resp_file == "":
            prompt_consumer.free_memory()

    return models_score
