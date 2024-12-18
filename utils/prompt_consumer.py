import transformers, torch,gc, warnings
from transformers.utils import ModelOutput
from tqdm import tqdm
from abc import ABC, abstractmethod
from .prompt_manager import GenericPromptManager

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
