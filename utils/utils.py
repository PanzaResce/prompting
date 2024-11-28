import requests, transformers, json, torch, re, os, gc
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import classification_report
from datasets import Dataset
from abc import ABC, abstractmethod
from utils.config import *

TOKEN= "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"
HEADERS = {"Authorization": f"Bearer {TOKEN}", 'Cache-Control': 'no-cache'}

def extract_model_name(card):
    return card.split("/")[-1]

def get_bnb_config(type: str):
    admissable_quant = ("8", "4", "None")
    if type not in admissable_quant:
        print(f"Quantization type must be one of {admissable_quant}, {type} not valid")
    
    if type == "8":
        return transformers.BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif type == "4":
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        return

def load_dataset(path, split):
    ds = load_from_disk(path)
    return ds[split]

def load_model_and_tokenizer(model_card, bnb_config):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.generation_config.top_p=1.0
    model.generation_config.temperature=1
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_card)
    return model, tokenizer

def create_pipeline(model, tokenizer):
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    # For older transformers versions
    pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'
    return pipe

def generate_responses(pipe, ds_test, base_prompt, temp, debug=0):
    if isinstance(base_prompt, MultiPrompt):
        return multiprompt_responses(pipe, ds_test, base_prompt, temp, debug)
    
    if debug > 0:
        print(f"Using the following prompt format: {base_prompt}")

    responses = []
    for clause in tqdm(ds_test["text"], total=len(ds_test)):
        messages = [{"role": "user", "content": base_prompt + clause}]
        
        gen_out = pipe(messages, max_new_tokens=30, temperature=temp, do_sample=False, return_full_text=False)
        responses.append(gen_out[0]["generated_text"].strip())
    return responses

def multiprompt_responses(pipe, ds_test, multiprompt, temp, debug=0):
    responses = []
    raw_responses = []
    if debug > 0:
        print("Using MultiPrompt")
    for clause in tqdm(ds_test["text"], total=len(ds_test)):
        clause_resp = []
        per_category_prompt = multiprompt.get_prompts(clause, debug)

        if debug > 1:
            print(f"Prompt used: \n {per_category_prompt}")

        messages = [[{"role": "user", "content": prompt}] for cat, prompt in per_category_prompt.items()]
        gen_out = pipe(messages, batch_size=4, max_new_tokens=5, temperature=temp, do_sample=False, return_full_text=False)
        # print(gen_out)
        for resp, cat in zip(gen_out, per_category_prompt.keys()):
            # if debug > 1:
            #     print(f"Response: {resp[-1]["generated_text"]}")
            raw_responses.append(resp)
            if resp[-1]["generated_text"].strip()[0] == "y":
                clause_resp.append(cat)
        
        if clause_resp == []:
            clause_resp.append("fair")
        
        responses.append(clause_resp)
    
    if debug > 0:
        return responses, raw_responses
    else:
        return responses

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
            true_sample[t] = 1

        pred_sample = [0] * len(label_to_id)
        if len(resp_tags) <= len(label_to_id):
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

        if debug > 2:
            print(f"Original response: {r}")

        y_true.append(true_sample)
        y_pred.append(pred_sample)

    report = classification_report(y_true, y_pred, zero_division=0, target_names=label_to_id.keys(), output_dict=True)
    return {"report": report}

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

def get_base_prompt(prompt_type, num_shots=None):
    if prompt_type == "zero":
        base_prompt = ZERO_PROMPT_1
    elif prompt_type == "zero_old":
        base_prompt = ZERO_PROMPT_0
    # elif prompt_type == "multi_few_half":
    #     base_prompt = FewMultiPrompt(MULTI_PROMPT_FEW, LABEL_TO_ID, num_shots=2, only_unfair_examples=False)
    elif prompt_type == "multi_few":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        base_prompt = FewMultiPrompt(MULTI_PROMPT_FEW, LABEL_TO_ID, per_category_examples, num_shots=num_shots, only_unfair_examples=True)
    elif prompt_type == "multi_zero":
        base_prompt = ZeroMultiPrompt(MULTI_PROMPT, LABEL_TO_ID)
    
    return base_prompt

def evaluate_models(endpoints, ds_test, ds_val, prompt_type, quant, num_shots, debug=0):
    label_to_id = LABEL_TO_ID
    bnb_config = get_bnb_config(quant)
    models_score = {}

    base_prompt = get_base_prompt(prompt_type, num_shots)

    for model_name, model_card in endpoints.items():
        print(f"------------ Evaluating model {model_name} on {prompt_type} with {quant} quantization ------------")
        model, tokenizer = load_model_and_tokenizer(model_card, bnb_config)
        pipe = create_pipeline(model, tokenizer)
        # Evaluate on test dataset
        test_responses = generate_responses(pipe, ds_test, base_prompt, temp=0, debug=debug)
        test_report = compute_f1_score(test_responses, ds_test["labels"], label_to_id, debug)

        if ds_val != None:
            # Evaluate on validation dataset
            validation_responses = generate_responses(pipe, ds_val, base_prompt, temp=0)
            validation_report = compute_f1_score(validation_responses, ds_test["labels"], label_to_id)
        else:
            validation_report = {"report": {}}

        models_score[model_name] = {
            "test": test_report,
            "validation": validation_report
        }

        output_dir = prompt_type+"_"+str(num_shots) if prompt_type in ["multi_few"] else prompt_type
        write_res_to_file(model_name, {model_name: {"test": test_report,"validation": validation_report}}, output_dir)
        
        # Free memory
        del model, pipe, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return models_score

def api_query(endpoint, payload, debug=0):
    """
    Send a POST request to the specified API endpoint with the given payload.

    Args:
        endpoint (str): The URL of the API endpoint to send the request to.
        payload (dict): The JSON payload to include in the POST request.

    Returns:
        dict: The JSON response from the API as a dictionary.

    Raises:
        SystemExit: If a request exception occurs, the function raises a SystemExit with the exception message.
    """

    if debug == 1:
        print(f"Running query on model {extract_model_name(endpoint)}")
    elif debug == 2:
        print(f"Running query on model {extract_model_name(endpoint)}\n Payload = {payload}")


    try:
        response = requests.post(endpoint, headers=HEADERS, json=payload)
    except requests.exceptions.RequestException as e: 
        raise SystemExit(e)
    return response.json()


class MultiPrompt(ABC):
    def __init__(self, prompt_template, label_to_id) -> None:
        self.prompt_template = prompt_template
        self.category_definitions = {k:"" for k in label_to_id.keys() if k != "fair"}
        self.set_descriptions()

    def set_descriptions(self):
        # New Definitions
        self.category_definitions["a"] = "A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion."
        self.category_definitions["ch"] = "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service"
        self.category_definitions["cr"] = "A clause is unfair if it gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content"
        self.category_definitions["j"] = "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)"
        self.category_definitions["law"] = "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence."
        self.category_definitions["ltd"] = "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law.' Such a clause is unfair unless it pertains to a force majeure case"
        self.category_definitions["ter"] = "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice"
        self.category_definitions["use"] = "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website"
        self.category_definitions["pinc"] = "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"
        
        # Old Definitions
        # self.category_definitions["j"] = "The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses stating that any judicial proceeding takes a residence away (i.e., in a different city, different country) are unfair."
        # self.category_definitions["law"] = "The choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. In every case where the clause defines the applicable law as the law of the consumer’s country of residence, it is considered as unfair."
        # self.category_definitions["ltd"] = "The limitation of liability clause stipulates that the duty to pay damages is limited or excluded for certain kinds of losses under certain conditions. Clauses that reduce, limit, or exclude the liability of the service provider are marked as unfair."
        # self.category_definitions["ch"] = "The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clauses are always considered as unfair."
        # self.category_definitions["ter"] = "The unilateral termination clause gives the provider the right to suspend and/or terminate the service and/or the contract and sometimes details the circumstances under which the provider claims to have a right to do so. Unilateral termination clauses that specify reasons for termination are marked as unfair. Clauses stipulating that the service provider may suspend or terminate the service at any time for any or no reasons and/or without notice are marked as unfair."
        # self.category_definitions["use"] = "The contract by using clause stipulates that the consumer is bound by the terms of use of a specific service simply by using the service, without even being required to mark that they have read and accepted them. These clauses are marked as unfair."
        # self.category_definitions["cr"] = "The content removal clause gives the provider a right to modify/delete user’s content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so. Clauses that indicate conditions for content removal are marked as unfair. Clauses stipulating that the service provider may remove content at their full discretion, and/or at any time for any or no reasons and/or without notice nor the possibility to retrieve the content, are marked as clearly unfair."
        # self.category_definitions["a"] = "The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process before the case could go to court. Clauses stipulating that the arbitration should take place in a state other than the state of the consumer’s residence and/or be based not on law but on the arbiter’s discretion are marked as unfair. Clauses defining arbitration as fully optional would be marked as fair."
        # self.category_definitions["pinc"] = "Identify clauses stating that consumers consent to the privacy policy simply by using the service. Such clauses are considered unfair."


    @abstractmethod
    def get_prompts(self, clause):
        pass

class ZeroMultiPrompt(MultiPrompt):
    def __init__(self, prompt_template, label_to_id):
        super().__init__(prompt_template, label_to_id)
        # self.prompt_template = prompt_template


    def get_prompts(self, clause):
        dict_of_prompts = {}
        for cat, definition in self.category_definitions.items():
            dict_of_prompts[cat] = self.prompt_template.format(cat_descr=definition, clause=clause)
        return dict_of_prompts

class FewMultiPrompt(MultiPrompt):
    def __init__(self, prompt_template, label_to_id, per_category_examples, num_shots, only_unfair_examples):
        super().__init__(prompt_template, label_to_id)

        self.per_category_examples = per_category_examples
        self.only_unfair_examples = only_unfair_examples
        self.num_shots = num_shots

    def get_prompts(self, clause, debug=0):
        dict_of_prompts = {}
        
        for category in self.category_definitions.keys():
            category_definition = self.category_definitions[category]

            if self.only_unfair_examples:
                examples = "\n".join([f"Clause: {ex}\nResponse: y" for ex in self.per_category_examples[category][:self.num_shots]])
            else:
                # first positive, second negative
                # examples = "\n".join([f"Clause: {self.per_category_examples[category][0]}\n Response: y", f"Clause: {self.per_category_examples[category][1]}\n Response: n"])
                raise NotImplementedError()

            dict_of_prompts[category] = self.prompt_template.format(cat_descr=category_definition, examples=examples, clause=clause)

            if debug > 0:
                print(f"EXAMPLES: \n {examples}")
                print(f"DEFINITION: \n {category_definition}")
                print(f"PROMPT: \n {dict_of_prompts[category]}")

        return dict_of_prompts

# python prompt_test.py -type multi_few -all -quant None -num_shots 2 > few_2 2>&1; python pretty_print_report.py -type multi_few_2; python prompt_test.py -type multi_few -all -quant None -num_shots 3 > few_3 2>&1; python pretty_print_report.py -type multi_few_3; python prompt_test.py -type multi_few -all -quant None -num_shots 4 > few_4 2>&1; python pretty_print_report.py -type multi_few_4