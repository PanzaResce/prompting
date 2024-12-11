import transformers, json, torch, re, os, gc, warnings
from transformers.utils import ModelOutput
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod
from utils.config import *

TOKEN= "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"
HEADERS = {"Authorization": f"Bearer {TOKEN}", 'Cache-Control': 'no-cache'}

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

def load_dataset(path, split):
    ds = load_from_disk(path)
    return ds[split]

def pick_only_unfair_clause(ds_test, num_elements=None):    
    new_ds_test = {"text": [], "labels": []}
    for clause, label in zip(ds_test["text"], ds_test["labels"]):
        if 0 not in label:
            new_ds_test["text"].append(clause)
            new_ds_test["labels"].append(label)

    if num_elements is None:
        num_elements = len(new_ds_test["text"])

    return {"text": new_ds_test["text"][:num_elements], "labels": new_ds_test["labels"][:num_elements]}

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


def compute_f1_score(responses, labels, label_to_id, unfair_only, debug=0):
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
            if unfair_only:
                true_sample[t-1] = 1
            else:
                true_sample[t] = 1

        pred_sample = [0] * len(label_to_id)
        for t in resp_tags:
            if t in label_to_id.keys():
                pred_sample[label_to_id[t]] = 1
            else:
                if debug > 2:
                    print(f"'{t}' is not a valid tag ({list(label_to_id.keys())})")

        print(f"{true_sample=}")
        print(f"{pred_sample=}")
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


def get_prompt_manager(prompt_type, label_to_id, num_shots=None):
    if prompt_type == "zero":
        prompt_manager = StandardPrompt(ZERO_PROMPT_1, max_new_tokens=30)
    elif prompt_type == "zero_old":
        prompt_manager = StandardPrompt(ZERO_PROMPT_0, max_new_tokens=30)
    # elif prompt_type == "multi_few_half":
    #     prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW, label_to_id, num_shots=2, only_unfair_examples=False)
    elif prompt_type == "multi_few":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW, label_to_id, per_category_examples, num_shots=num_shots, only_unfair_examples=True, max_new_tokens=2)
        prompt_manager.set_response_type(positive="y", negative="n")
    elif prompt_type == "multi_few_no_templ":
        per_category_examples = read_examples_from_json(exclude_fair=True)
        prompt_manager = FewMultiPrompt(MULTI_PROMPT_FEW_NO_TEMPL, label_to_id, per_category_examples, num_shots=num_shots, only_unfair_examples=True, max_new_tokens=2)
        prompt_manager.set_response_type(positive="Yes", negative="No")
        # prompt_manager.set_response_type(positive="y", negative="n")
    elif prompt_type == "multi_zero":
        prompt_manager = ZeroMultiPrompt(MULTI_PROMPT, label_to_id, max_new_tokens=2)
        prompt_manager.set_response_type(positive="y", negative="n")
    
    return prompt_manager


def factory_instantiate_prompt_consumer(prompt_type, model_card, bnb_config, num_shots, label_to_id):
    model, tokenizer = load_model_and_tokenizer(model_card, bnb_config)
    model_has_chat_template = not tokenizer.chat_template is None
    prompt_manager = get_prompt_manager(prompt_type, label_to_id, num_shots)

    if prompt_type == "multi_few_no_templ" or not model_has_chat_template:
        return BareModelPromptConsumer(prompt_manager, model, tokenizer)
        # return NoTemplatePromptConsumer(prompt_manager, model, tokenizer)
    else:
        return PipelinePromptConsumer(prompt_manager, model, tokenizer)


def evaluate_models(endpoints, ds_test, ds_val, prompt_type, quant, num_shots, write_to_file, unfair_only, debug=0):
    label_to_id = LABELS.labels_to_id(unfair_only)
    print(label_to_id)

    bnb_config = get_bnb_config(quant)
    models_score = {}

    # prompt_manager = get_prompt_manager(prompt_type, num_shots)

    for model_name, model_card in endpoints.items():
        print(f"------------ Evaluating model {model_name} on {prompt_type} with {quant} bit quantization ------------")

        prompt_consumer = factory_instantiate_prompt_consumer(prompt_type, model_card, bnb_config, num_shots, label_to_id)
        test_responses = prompt_consumer.generate_responses(ds_test, debug)
        test_report = compute_f1_score(test_responses, ds_test["labels"], label_to_id, unfair_only, debug)

        validation_report = {"report": {}}

        models_score[model_name] = {
            "test": test_report,
            "validation": validation_report
        }

        if write_to_file:
            output_dir = prompt_type+"_"+str(num_shots) if prompt_type in ["multi_few"] else prompt_type
            write_res_to_file(model_name, {model_name: {"test": test_report,"validation": validation_report}}, output_dir)

        prompt_consumer.free_memory()

    return models_score


class GenericPromptManager(ABC):
    def __init__(self, prompt_template, max_new_tokens):
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
    
    @abstractmethod
    def get_prompts(self, clause) -> list:
        pass

    @abstractmethod
    def format_response(self, clean_response):
        pass

class StandardPrompt(GenericPromptManager):
    def get_prompts(self, clause):
        return [self.prompt_template + clause]

    def format_response(self, clean_response):
        # clean_response = raw_response[0][0]["generated_text"].strip()
        return clean_response

class MultiPrompt(GenericPromptManager, ABC):
    def __init__(self, prompt_template, label_to_id, max_new_tokens) -> None:
        super().__init__(prompt_template, max_new_tokens)

        self.category_definitions = {k:"" for k in label_to_id.keys() if k != "fair"}
        self.set_descriptions()

    def set_descriptions(self):
        # New Definitions
        self.category_definitions["a"] = "A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion."
        self.category_definitions["ch"] = "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service"
        self.category_definitions["cr"] = "A clause is unfair if it gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content"
        self.category_definitions["j"] = "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)"
        self.category_definitions["law"] = "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence."
        self.category_definitions["ltd"] = "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law'. Such a clause is unfair unless it pertains to a force majeure case"
        self.category_definitions["ter"] = "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice"
        self.category_definitions["use"] = "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website"
        self.category_definitions["pinc"] = "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"

    def set_response_type(self, positive, negative):
        self.response_type = {"positive": positive, "negative": negative}

    @property
    def positive_response(self):
        return self.response_type["positive"]

    @property
    def negative_response(self):
        return self.response_type["negative"]

    def format_response(self, clean_response):
        # clean_resp = [generated[-1]["generated_text"].strip() for generated in raw_response]
        clause_resp = []

        for resp, cat in zip(clean_response, self.category_definitions.keys()):
            if resp == self.positive_response:
                clause_resp.append(cat)
            elif resp != self.negative_response:
                print("QUA", resp)
        
        if clause_resp == []:
            clause_resp.append("fair")

        return clause_resp     

class ZeroMultiPrompt(MultiPrompt):

    def get_prompts(self, clause, debug=0):
        list_of_prompts = []
        for definition in self.category_definitions.values():
            formatted_prompt = self.prompt_template.format(cat_descr=definition, clause=clause)
            list_of_prompts.append(formatted_prompt)
            if debug > 0:
                print(f"{definition=}")
                print(f"{formatted_prompt=}")
        return list_of_prompts

class FewMultiPrompt(MultiPrompt):
    def __init__(self, prompt_template, label_to_id, per_category_examples, num_shots, only_unfair_examples, max_new_tokens):
        super().__init__(prompt_template, label_to_id, max_new_tokens)

        self.per_category_examples = per_category_examples
        self.only_unfair_examples = only_unfair_examples
        self.num_shots = num_shots
    
    def get_prompts(self, clause, debug=0):
        list_of_prompts = []
        
        for category in self.category_definitions.keys():
            category_definition = self.category_definitions[category]

            if self.only_unfair_examples:
                examples = "\n".join([f"Clause: {ex}\nResponse: {self.positive_response}" for ex in self.per_category_examples[category][:self.num_shots]])
            else:
                # first positive, second negative
                # examples = "\n".join([f"Clause: {self.per_category_examples[category][0]}\n Response: y", f"Clause: {self.per_category_examples[category][1]}\n Response: n"])
                raise NotImplementedError()

            formatted_prompt = self.prompt_template.format(cat_descr=category_definition, examples=examples, clause=clause)
            list_of_prompts.append(formatted_prompt)

            if debug > 1:
                print(f"EXAMPLES: \n {examples}")
                print(f"DEFINITION: \n {category_definition}")
                print(f"PROMPT for category '{category}': \n {formatted_prompt}")

        return list_of_prompts

class PromptConsumer(ABC):
    def __init__(self, prompt_manager: GenericPromptManager, model, tokenizer):
        self.prompt_manager = prompt_manager
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_responses(self, dataset, debug=0):
        responses = []
        for clause in tqdm(dataset["text"], total=len(dataset)):

            model_input = self.format_input(clause, debug)
            raw_output = self.run_model(model_input)
            response = self.format_response(raw_output)

            if debug > 1:
                print(f"{model_input=}")
                print(f"{raw_output=}")
                print(f"{response=}")
                
            responses.append(response)

        return responses

    @abstractmethod
    def format_input(self, clause):
        pass

    @abstractmethod
    def run_model(self, model_input):
        pass
    
    @abstractmethod
    def format_response(self, raw_output):
        pass

    def free_memory(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

class PipelinePromptConsumer(PromptConsumer):
    def __init__(self, prompt_manager, model, tokenizer):
        super().__init__(prompt_manager, model, tokenizer)

        self.pipeline = self.create_pipeline()
    
    def create_pipeline(self):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        pipeline.model.generation_config.do_sample = False

        # For older transformers versions
        pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        pipeline.tokenizer.padding_side = 'left'
        return pipeline

    def format_input(self, clause, debug=0):
        prompt_list = self.prompt_manager.get_prompts(clause, debug)
        messages = [[{"role": "user", "content": prompt}] for prompt in prompt_list]
        
        if debug > 0:
            print(f"{prompt_list=}")
            print(f"{messages=}")

        return messages

    def run_model(self, model_input):
        raw_output = self.pipeline(model_input, batch_size=4, max_new_tokens=self.prompt_manager.max_new_tokens, temperature=1, do_sample=False, return_full_text=False)
        return raw_output

    def format_response(self, raw_output):
        # clean_response = raw_response[0][0]["generated_text"].strip()
        if len(raw_output) > 1:
            clean_response = [generated[-1]["generated_text"].strip() for generated in raw_output]
        else:
            clean_response = raw_output[0][0]["generated_text"].strip()
        
        return self.prompt_manager.format_response(clean_response)
    
    def free_memory(self):
        del self.pipeline
        super().free_memory()


class NoTemplatePromptConsumer(PipelinePromptConsumer):
    def __init__(self, prompt_manager, model, tokenizer):
        super().__init__(prompt_manager, model, tokenizer)
        if not tokenizer.chat_template is None:
            warnings.warn("Using a NoTemplatePromptConsumer for a model that has a template")

    def format_input(self, clause, debug=0):
        return self.prompt_manager.get_prompts(clause, debug)


# TODO: Should modify this class to handle both few and zero shot 
class BareModelPromptConsumer(PromptConsumer):
    def __init__(self, prompt_manager, model, tokenizer):
        super().__init__(prompt_manager, model, tokenizer)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.ids_to_words = {v:k for k,v in self.tokenizer.vocab.items()}


        self.model.generation_config.max_new_tokens = 2
        self.model.generation_config.temperature = 1
        self.model.generation_config.output_scores = True
        self.model.generation_config.return_dict_in_generate = True
        self.model.generation_config.do_sample = False
        self.model.generation_config.min_new_tokens = 2

    def format_input(self, clause, debug):
        prompt_list = self.prompt_manager.get_prompts(clause, debug)
        input_ids = self.tokenizer(prompt_list, return_tensors="pt", padding=True).to("cuda")
        return input_ids
    
    def run_model(self, model_input) -> ModelOutput:
        ret_tensor = self.model.generate(**model_input)
        return ret_tensor
    
    def format_response(self, raw_output):
        clean_resp = []

        for resp in raw_output["scores"][0]:
            # print([self.ids_to_words[index.item()] for index in torch.topk(resp, 5)[1]])

            positive_response_score = resp[self.tokenizer.vocab[self.prompt_manager.positive_response]]
            negative_response_score = resp[self.tokenizer.vocab[self.prompt_manager.negative_response]]
            
            if positive_response_score > negative_response_score:
                clean_resp.append(self.prompt_manager.positive_response)
            else:
                clean_resp.append(self.prompt_manager.negative_response)
        
        # print(clean_resp)
        return self.prompt_manager.format_response(clean_resp)
