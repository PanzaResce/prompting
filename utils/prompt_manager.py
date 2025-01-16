import warnings
from abc import ABC, abstractmethod

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
    def get_prompts(self, clause, debug=0):
        return [self.prompt_template + clause]

    def format_response(self, clean_response):
        # clean_response = raw_response[0][0]["generated_text"].strip()
        return clean_response

class MultiPrompt(GenericPromptManager, ABC):
    def __init__(self, prompt_template, label_to_id, max_new_tokens) -> None:
        super().__init__(prompt_template, max_new_tokens)

        self.category_definitions = {k:"" for k in label_to_id.keys() if k != "fair"}
        self.init_definitions()

    def init_definitions(self, definitions_dict=None):
        if definitions_dict == None:
            self.category_definitions["a"] = "A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion."
            self.category_definitions["ch"] = "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service"
            self.category_definitions["cr"] = "A clause is unfair if it gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content"
            self.category_definitions["j"] = "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)"
            self.category_definitions["law"] = "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence."
            self.category_definitions["ltd"] = "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law'. Such a clause is unfair unless it pertains to a force majeure case"
            self.category_definitions["ter"] = "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice"
            self.category_definitions["use"] = "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website"
            self.category_definitions["pinc"] = "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"
            
            # Old
            # self.category_definitions["j"] = "The jurisdiction clause specifies what courts have the competence to adjudicate disputes. A clause is unfair whenever it states that judicial proceedings take place in a location different from the consumer's place of residence (e.g., in a different city or country)."
            # self.category_definitions["law"] = "The choice of law clause specifies what law will govern the contract and be applied in potential disputes. A clause is unfair whenever it states that the applicable law is different from the law of the consumer’s place of residence."
            # self.category_definitions["ltd"] = "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or include a blanket phrase like 'to the fullest extent permissible by law.' Such a clause is always unfair, unless it pertains to a force majeure case."
            # self.category_definitions["ch"] = "The unilateral change clause specifies if and under what conditions the provider can unilaterally change and modify the contract and/or the service. Such a clause is always unfair."
            # self.category_definitions["ter"] = "The unilateral termination clause states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for certain reasons, or at any time, for any or no reason, with or without notice. Such a clause is always unfair."
            # self.category_definitions["use"] = "The contract by using clause states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website. Such a clause is always unfair."
            # self.category_definitions["cr"] = "The content removal clause gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility of retrieving the content. Such a clause is always unfair."
            # self.category_definitions["a"] = "The arbitration clause requires or allows the parties to resolve their disputes through arbitration before the case can go to court. A clause is unfair whenever the arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion."
            # self.category_definitions["pinc"] = "The privacy included clause identifies clauses that (a) explicitly state that, by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or (b) incorporate the privacy policy into the terms, particularly if preceded by a contract by using clause. Such clauses are always unfair."
        else:
            # Init from dict
            for k in self.category_definitions.keys():
                self.category_definitions[k] = definitions_dict[k]
    
    def set_response_type(self, positive, negative):
        self.response_type = {"positive": positive, "negative": negative}

    @property
    def positive_response(self):
        return self.response_type["positive"]

    @property
    def negative_response(self):
        return self.response_type["negative"]

    def isResponsePositive(self, response):
        return response == self.positive_response

    def isResponseNegative(self, response):
        return response == self.negative_response

    def format_response(self, clean_response):
        clause_resp = []

        for resp, cat in zip(clean_response, self.category_definitions.keys()):
            if self.isResponsePositive(resp):
                clause_resp.append(cat)
            elif not self.isResponseNegative(resp):
                warnings.warn(f"Response '{resp}' different from both positive '{self.positive_response}' and negative '{self.negative_response}' response.")

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
                examples = "\n".join([f"Clause: {ex}\nResponse: {self.positive_response}" 
                                      for ex in self.per_category_examples[category][:self.num_shots]])
            else:
                # first positive, second negative
                # examples = "\n".join([f"Clause: {self.per_category_examples[category][0]}\n Response: y", f"Clause: {self.per_category_examples[category][1]}\n Response: n"])
                raise NotImplementedError()

            formatted_prompt = self.prompt_template.format(cat_descr=category_definition, examples=examples, clause=clause)
            list_of_prompts.append(formatted_prompt)

            if debug == 1:
                print(f"EXAMPLES: \n{examples}")
                print(f"DEFINITION: \n{category_definition}")
                print(f"PROMPT for category '{category}': \n{formatted_prompt}")

        return list_of_prompts