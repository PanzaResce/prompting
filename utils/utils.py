import requests, transformers, json, torch, re, warnings
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import f1_score

from utils.config import *

TOKEN= "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"
HEADERS = {"Authorization": f"Bearer {TOKEN}", 'Cache-Control': 'no-cache'}

def extract_model_name(card):
    return card.split("/")[-1]

def get_bnb_config():
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

def generate_responses(pipe, ds_test, base_prompt, temp):
    responses = []
    for clause, label in tqdm(zip(ds_test["text"], ds_test["labels"]), total=len(ds_test)):
        messages = [{"role": "system", "content": base_prompt},
                    {"role": "user", "content": clause}]
        gen_out = pipe(messages, max_new_tokens=50, temperature=temp, do_sample=False)[0]
        gen_out["labels"] = label
        responses.append(gen_out)
    return responses

def compute_f1_score(responses, label_to_id, debug=0):
    y_true, y_pred = [], []
    for r in responses:
        # resp_tags = [t for t in r["generated_text"][2]["content"].replace("<", "").replace(">", "").replace(",", "").split(" ")]
        resp_tags = re.findall(r"(?<=<)[^>]+(?=>)", r["generated_text"][2]["content"])


        true_sample = [0] * len(label_to_id)
        for t in r["labels"]:
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
            print(f"Original response: {r['generated_text'][2]['content']}")

        y_true.append(true_sample)
        y_pred.append(pred_sample)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return {"macro": macro_f1, "micro": micro_f1}

def write_res_to_file(model_name, model_score, prompt_type):
    with open(f'out/{prompt_type}/{model_name}_score.json', 'w') as file: 
        file.write(json.dumps(model_score))

def evaluate_models(endpoints, ds_test, ds_val, prompt_type):
    label_to_id = LABEL_TO_ID
    bnb_config = get_bnb_config()
    models_score = {}

    if prompt_type == "zero":
        base_prompt = ZERO_PROMPT
    elif prompt_type == "few":
        base_prompt = FEW_PROMPT

    for name, model_card in endpoints.items():
        print(f"------------ Evaluating model {name} on {prompt_type} ------------")
        model, tokenizer = load_model_and_tokenizer(model_card, bnb_config)
        pipe = create_pipeline(model, tokenizer)

        # Evaluate on test dataset
        test_responses = generate_responses(pipe, ds_test, base_prompt)
        test_scores = compute_f1_score(test_responses, label_to_id)

        if ds_val != None:
            # Evaluate on validation dataset
            validation_responses = generate_responses(pipe, ds_val, base_prompt)
            validation_scores = compute_f1_score(validation_responses, label_to_id)
        else:
            validation_scores = {"macro": 0.0, "micro": 0.0}

        models_score[name] = {
            "test": test_scores,
            "validation": validation_scores
        }

        write_res_to_file(name, {name: {"test": test_scores,"validation": validation_scores}}, prompt_type)

    return models_score

# def evaluate_model(model, sys_prompt, dataset):
#     prompt_template = f"""
#     <|system|>
#     {sys_prompt}
#     <|user|>
#     {{}}
#     <|assistant|>
#     """

#     resp_list = []
#     for el, label in zip(dataset["text"], dataset["labels"]):
#         prompt = prompt_template.format(el)
#         payload = {
#             "inputs":f"{prompt}",
#             "parameters": {
#                 "return_full_text": False,
#                 "max_new_tokens": 300,
#                 "do_sample": False
#             },
#             "options": {
#                 "use_cache": False,
#             },
#         }

#         resp_list.append((api_query(model, payload), label))
#     return resp_list

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