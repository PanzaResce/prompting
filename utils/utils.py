import requests, transformers, json, torch, re, warnings, os, gc
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import f1_score
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
        return multiprompt_responses(pipe, ds_test, base_prompt, temp, debug=0)
    
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
    if debug > 0:
        print("Using MultiPrompt")
    for clause in tqdm(ds_test["text"], total=len(ds_test)):
        clause_resp = []
        prompt_dict = multiprompt.get_prompts(clause)

        if debug > 1:
            print(f"Prompt used: \n {prompt_dict}")

        messages = [[{"role": "user", "content": prompt}] for cat, prompt in prompt_dict.items()]
        gen_out = pipe(messages, batch_size=4, max_new_tokens=5, temperature=temp, do_sample=False, return_full_text=False)
        # print(gen_out)
        for resp, cat in zip(gen_out, prompt_dict.keys()):
            # if debug > 1:
            #     print(f"Response: {resp[-1]["generated_text"]}")
            if resp[-1]["generated_text"].strip() == "y":
                clause_resp.append(cat)
        
        if clause_resp == []:
            clause_resp.append("fair")
        
        responses.append(clause_resp)
    
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

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return {"macro": macro_f1, "micro": micro_f1}

def write_res_to_file(model_name, model_score, prompt_type):
    os.makedirs(f"out/{prompt_type}/", exist_ok=True)
    with open(f'out/{prompt_type}/{model_name}_score.json', 'w') as file: 
        file.write(json.dumps(model_score))

def evaluate_models(endpoints, ds_test, ds_val, prompt_type, quant):
    label_to_id = LABEL_TO_ID
    bnb_config = get_bnb_config(quant)
    models_score = {}

    if prompt_type == "zero":
        base_prompt = ZERO_PROMPT_1
    elif prompt_type == "zero_old":
        base_prompt = ZERO_PROMPT_0
    elif prompt_type == "multi_few_half":
        base_prompt = FewMultiPrompt(MULTI_PROMPT_FEW, LABEL_TO_ID, FEW_EXAMPLES_HALF, 2)
    elif prompt_type == "multi_few":
        base_prompt = FewMultiPrompt(MULTI_PROMPT_FEW, LABEL_TO_ID, FEW_EXAMPLES_POS, 1)
    elif prompt_type == "multi_zero":
        base_prompt = ZeroMultiPrompt(MULTI_PROMPT, LABEL_TO_ID)

    for name, model_card in endpoints.items():
        print(f"------------ Evaluating model {name} on {prompt_type} with {quant} quantization ------------")
        model, tokenizer = load_model_and_tokenizer(model_card, bnb_config)
        pipe = create_pipeline(model, tokenizer)
        # Evaluate on test dataset
        test_responses = generate_responses(pipe, ds_test, base_prompt, temp=0)
        test_scores = compute_f1_score(test_responses, ds_test["labels"], label_to_id)

        if ds_val != None:
            # Evaluate on validation dataset
            validation_responses = generate_responses(pipe, ds_val, base_prompt, temp=0)
            validation_scores = compute_f1_score(validation_responses, ds_test["labels"], label_to_id)
        else:
            validation_scores = {"macro": 0.0, "micro": 0.0}

        models_score[name] = {
            "test": test_scores,
            "validation": validation_scores
        }

        write_res_to_file(name, {name: {"test": test_scores,"validation": validation_scores}}, prompt_type)
        
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
    def __init__(self, base_prompt, label_to_id) -> None:
        self.base_prompt = base_prompt
        self.cat_definition = {k:"" for k in label_to_id.keys() if k != "fair"}
        self.set_descriptions()

    def set_descriptions(self):
        # New Definitions
        self.cat_definition["a"] = "A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion."
        self.cat_definition["ch"] = "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service"
        self.cat_definition["cr"] = "A clause is unfair if it gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content"
        self.cat_definition["j"] = "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)"
        self.cat_definition["law"] = "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence."
        self.cat_definition["ltd"] = "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law.' Such a clause is unfair unless it pertains to a force majeure case"
        self.cat_definition["ter"] = "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice"
        self.cat_definition["use"] = "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website"
        self.cat_definition["pinc"] = "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"
        
        # Old Definitions
        # self.cat_definition["j"] = "The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses stating that any judicial proceeding takes a residence away (i.e., in a different city, different country) are unfair."
        # self.cat_definition["law"] = "The choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. In every case where the clause defines the applicable law as the law of the consumer’s country of residence, it is considered as unfair."
        # self.cat_definition["ltd"] = "The limitation of liability clause stipulates that the duty to pay damages is limited or excluded for certain kinds of losses under certain conditions. Clauses that reduce, limit, or exclude the liability of the service provider are marked as unfair."
        # self.cat_definition["ch"] = "The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clauses are always considered as unfair."
        # self.cat_definition["ter"] = "The unilateral termination clause gives the provider the right to suspend and/or terminate the service and/or the contract and sometimes details the circumstances under which the provider claims to have a right to do so. Unilateral termination clauses that specify reasons for termination are marked as unfair. Clauses stipulating that the service provider may suspend or terminate the service at any time for any or no reasons and/or without notice are marked as unfair."
        # self.cat_definition["use"] = "The contract by using clause stipulates that the consumer is bound by the terms of use of a specific service simply by using the service, without even being required to mark that they have read and accepted them. These clauses are marked as unfair."
        # self.cat_definition["cr"] = "The content removal clause gives the provider a right to modify/delete user’s content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so. Clauses that indicate conditions for content removal are marked as unfair. Clauses stipulating that the service provider may remove content at their full discretion, and/or at any time for any or no reasons and/or without notice nor the possibility to retrieve the content, are marked as clearly unfair."
        # self.cat_definition["a"] = "The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process before the case could go to court. Clauses stipulating that the arbitration should take place in a state other than the state of the consumer’s residence and/or be based not on law but on the arbiter’s discretion are marked as unfair. Clauses defining arbitration as fully optional would be marked as fair."
        # self.cat_definition["pinc"] = "Identify clauses stating that consumers consent to the privacy policy simply by using the service. Such clauses are considered unfair."


    @abstractmethod
    def get_prompts(self, clause):
        pass

class ZeroMultiPrompt(MultiPrompt):
    def __init__(self, base_prompt, label_to_id):
        super().__init__(base_prompt, label_to_id)
        # self.base_prompt = base_prompt


    def get_prompts(self, clause):
        out_prompts = {}
        for cat, definition in self.cat_definition.items():
            out_prompts[cat] = self.base_prompt.format(cat_descr=definition, clause=clause)
        return out_prompts

class FewMultiPrompt(MultiPrompt):
    def __init__(self, base_prompt, label_to_id, examples_dict, num_shot):
        super().__init__(base_prompt, label_to_id)
        self.examples_dict = examples_dict
        self.num_shot = num_shot

    def get_prompts(self, clause, debug=0):
        out_prompts = {}
        
        for cat in self.cat_definition.keys():
            definition = self.cat_definition[cat]
            # we assume one positive examples
            examples = "\n".join([f"Clause: {ex}\n Response: y" for ex in self.examples_dict[cat][:self.num_shot]])

            # first positive, second negative
            # examples = "\n".join([f"Clause: {self.examples_dict[cat][0]}\n Response: y", f"Clause: {self.examples_dict[cat][1]}\n Response: n"])

            out_prompts[cat] = self.base_prompt.format(cat_descr=definition, examples=examples, clause=clause)

            if debug > 0:
                print(f"EXAMPLES: \n {examples}")
                print(f"DEFINITION: \n {definition}")
                print(f"PROMPT: \n {out_prompts[cat]}")

        return out_prompts
