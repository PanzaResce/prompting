import transformers, torch, gc, warnings, re
from transformers.utils import ModelOutput
from tqdm import tqdm
from abc import ABC, abstractmethod
from .config import LABELS, CHAIN_0_DEF, CHAIN_1_DEF

class PromptConsumer(ABC):
    default_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% if loop.index0 == 0 %}<s>[INST] <<SYS>>{{ message['content'] | trim }}<</SYS>>{% elif message['role'] == 'user' %}{{ message['content'] | trim }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] | trim }} [/INST]{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' [/INST]' }}{% endif %}"

    def __init__(self, prompt_manager, model, tokenizer):
        self.prompt_manager = prompt_manager
        self.model = model
        self.tokenizer = tokenizer
    
    def handle_oom_wrapper(method):
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except RuntimeError as e:
                print(e)
                print("-------- CUDA MEMORY SUMMARY --------")
                print(torch.cuda.memory_summary())
            return None
        return wrapper

    @handle_oom_wrapper
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

    def clean_response(self, raw_output):
        def apply_cleaning(response):
            response = re.sub(r'[^a-zA-Z]', '', response)
            cleaned = response.strip().lower()
            return cleaned
        
        if len(raw_output) > 1:
            clean_response = [apply_cleaning(generated[-1]["generated_text"]) for generated in raw_output]
        else:
            clean_response = [apply_cleaning(raw_output[0][0]["generated_text"])]
        return clean_response
        # return [generated[-1]["generated_text"].strip().lower() for generated in raw_output]

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
        raw_output = self.pipeline(model_input, batch_size=1, max_new_tokens=self.prompt_manager.max_new_tokens, temperature=1, do_sample=False, return_full_text=False)
        return raw_output

    def format_response(self, raw_output):
        # if len(raw_output) > 1:
        #     clean_response = [generated[-1]["generated_text"].strip().lower() for generated in raw_output]
        # else:
        #     clean_response = raw_output[0][0]["generated_text"].strip().lower()
        clean_response = self.clean_response(raw_output)
        
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

class SVMPromptConsumer(PipelinePromptConsumer):
    def __init__(self, prompt_manager, model, tokenizer, svm, resp_file=""):
        self.svm = svm
        if resp_file != "":
            self.use_file = True
            self.true_labels, self.predicted_labels, self.clauses = self.read_responses_from_file(resp_file)
        else:
            super().__init__(prompt_manager, model, tokenizer)

    def read_responses_from_file(self, resp_file):
        true_labels = []
        predicted_labels = []
        clauses = []
        file_path = f"out/{resp_file}"

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    clause, true_label, predicted_label = parts[-3:]

                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)
                    clauses.append(clause)
                else:
                    print(parts)
        return true_labels, predicted_labels, clauses

    def generate_responses(self, dataset, debug=0):
        unfair_from_svm = self.svm.predict(dataset)
        unfair_from_svm = [1 if pred == 1 else 0 for pred in unfair_from_svm]

        responses = []
        index = 0
        print(f"Unfair clauses filtered by SVM: {sum(unfair_from_svm)}")
        for clause in tqdm(dataset["text"], total=len(dataset)):
            if unfair_from_svm[index] == 1:
                if self.use_file:
                    response = self.predicted_labels[index]
                    # if self.clauses[index] != clause:
                    #     print(self.clauses[index])
                    #     print(clause)
                else:
                    model_input = self.format_input(clause, debug)
                    raw_output = self.run_model(model_input)
                    response = self.format_response(raw_output)

                if debug > 0:
                    print(f"{model_input=}")
                    print(f"{raw_output=}")
                    print(f"{response=}")
            else:
                response = ["fair"]
                
            responses.append(response)
            index +=1

        return responses

class PromptChainConsumer(PipelinePromptConsumer):
    def __init__(self, prompt_manager, model, tokenizer):
        super().__init__(prompt_manager, model, tokenizer)
        self.prompt_manager[0].init_definitions(CHAIN_0_DEF)
        self.prompt_manager[1].init_definitions(CHAIN_1_DEF)
        self.filtered_responses = 0
        # self.prompt_manager[1].init_definitions({k:name for k, name in zip(CHAIN_1_DEF.keys(), LABELS.labels_full_name[1:])})

    def generate_responses(self, dataset, debug=0):
        responses = super().generate_responses(dataset, debug)
        print(f"The first prompt filtered {self.filtered_responses} responses")
        return responses
    
    def first_prompt_response_is_fair(self, first_response):
        if first_response[0] == "fair":
            return True
        return False

    def format_input(self, clause, debug=0):
        model_input_list = []
        for manager in self.prompt_manager:
            prompt_list = manager.get_prompts(clause)
            messages = [[{"role": "user", "content": prompt}] for prompt in prompt_list]

            model_input_list.append(messages)
        return model_input_list
    
    def run_model(self, model_input: list):
        raw_output_0 = self.pipeline(model_input[0], batch_size=1, max_new_tokens=self.prompt_manager[0].max_new_tokens, temperature=1, do_sample=False, return_full_text=False)
        torch.cuda.empty_cache()
        clean_output_0 = self.clean_response(raw_output_0)
        resp_0 = self.prompt_manager[0].format_response(clean_output_0)
        # print(f"{resp_0=}")
        # print(f"{raw_output_0=}")

        if self.first_prompt_response_is_fair(resp_0):
            final_resp = resp_0
            self.filtered_responses +=1
        else:
            index_to_keep = [i for cat in resp_0 for i in range(len(LABELS.get_unfair_labels())) if cat == LABELS.labels[i+1]]
            candidate_positive_clauses = [model_input[1][i] for i in index_to_keep]   # i-1 because we don't have the "fair" case in the input
            # print(f"{candidate_positive_clauses=}")
            raw_output_1 = self.pipeline(candidate_positive_clauses, batch_size=1, max_new_tokens=self.prompt_manager[1].max_new_tokens, temperature=1, do_sample=False, return_full_text=False)
            clean_output_1 = self.clean_response(raw_output_1)
            # print(f"{raw_output_1=}")
            # print(f"{clean_output_1=}")
            # print(f"{index_to_keep=}")

            categories = [self.prompt_manager[1].negative_response]*len(LABELS.get_unfair_labels())
            for index, resp in zip(index_to_keep, clean_output_1):
                categories[index] = resp
            # print(f"{categories=}")
            resp_1 = self.prompt_manager[1].format_response(categories)
            # print(f"{raw_output_1=}")
            # print(f"{resp_1=}")
            final_resp = resp_1

        # print(f"{final_resp=}")
        return final_resp

    def format_response(self, raw_output):
        return raw_output