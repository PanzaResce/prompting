ENDPOINTS = {
    # "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    # "phi3": "microsoft/Phi-3-medium-128k-instruct",
    # "codestral": "mistralai/Codestral-22B-v0.1",
    # "nemo": "mistralai/Mistral-Nemo-Instruct-2407",  
    # "llama8": "meta-llama/Meta-Llama-3-8B-Instruct",
    # "lawllm": "AdaptLLM/law-chat",
    # "gemma27": "google/gemma-2-27b-it",
    "gemma9": "google/gemma-2-9b-it",
    # "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
    # "qwen32": "Qwen/Qwen2.5-32B-Instruct"
}

ZERO_PROMPT = """
Your task is to classify clauses from Terms of Service documents to determine if they are unfair according to the following unfairness categories.
Categories of Unfair Clauses
  -Jurisdiction <j>: The jurisdiction clause specifies what courts have the competence to adjudicate disputes. A clause is (potentially) unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence).
  -Choice of Law <law>: The choice of law clause specifies what law will govern the contract and be applied in potential disputes. A clause is (potentially) unfair whenever it states that the applicable law is different from the law of the consumer's place of residence.
  -Limitation of Liability <ltd>: The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like "to the fullest extent permissible by law." Such a clause is always (potentially) unfair unless it pertains to a force majeure case.
  -Unilateral Change <ch>: The unilateral change clause specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service. Such a clause is always (potentially) unfair.
  -Unilateral Termination <ter>: The unilateral termination clause states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice. Such a clause is always (potentially) unfair.
  -Contract by Using <use>: The contract by using clause states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website. Such a clause is always (potentially) unfair.
  -Content Removal <cr>: The content removal clause gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content. Such a clause is always (potentially) unfair.
  -Arbitration <a>: The arbitration clause requires or allows the parties to resolve their disputes through arbitration before the case could go to court. A clause is (potentially) unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion.
  -Privacy Included <pinc>: This category includes clauses that (a) explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or (b) state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a "contract by using" clause. Such clauses are always (potentially) unfair.
A clause can be assigned to zero or more unfairness categories. If a clause is unfair respond only with the corresponding tags. If a clause is not unfair, respond only with "<fair>". Do not provide any additional explanations.
Clause:
"""

# MULTI_PROMPT = """Evaluate the following clause from a terms of service document based on the provided definition of unfairness:  
# Definition:  
# {cat_descr}  

# If the clause is unfair according to the definition respond with "y" otherwise respond with "n". Respond with "n" also if the clause is unrelated to the definition. Do not include any additional text or explanation.
# Clause: {clause}
# """

MULTI_PROMPT = """Evaluate the following clause based on the provided definition of unfairness:  
Definition:
{cat_descr}

If the clause is unfair according to the definition respond with 'yes' otherwise respond with 'no'.
Provide only 'yes' or 'no' as the output. Do not include any explanations or additional text.

Test Input:
Clause: {clause}
Response:"""


MULTI_PROMPT_FEW = """Evaluate the following clause from a terms of service based on the provided definition of unfairness:  
Definition:
{cat_descr}

If the clause is unfair according to the definition respond with 'yes' otherwise respond with 'no'.
Provide only 'yes' or 'no' as the output. Do not include any explanations or additional text.

Examples:
{examples}

Test Input:
Clause: {clause}
Response:"""

MULTI_PROMPT_FEW_NO_TEMPL = """You are an expert in analyzing legal text to identify whether a clause is unfair according to the following definition:
{cat_descr}

Here are some examples:
{examples}

Is the following clause unfair ?
Clause: {clause}
Response:"""

PROMPT_CHAIN_0 = """You are analyzing terms of service. Please, tell wether the Clause talks about the subject specified in the definition.
Definition: {cat_descr}
Please answer only with 'yes' or 'no'. Do not include any explanations or additional text.

Test Input:
Clause: {clause}
Response:"""

PROMPT_CHAIN_1 = """You are analyzing terms of service. Please, tell wether the input clause is unfair. {cat_descr}
If the clause is unfair respond with 'yes' otherwise respond with 'no'.
Provide only 'yes' or 'no' as the output. Do not include any explanations or additional text.

Examples:
{examples}

Test Input:
Clause: {clause}
Response:"""

class Labels():
    def __init__(self):
        # This order is important (?)
        self.labels = ["fair", "a", "ch", "cr", "j", "law", "ltd", "ter", "use", "pinc"]
        self.labels_full_name = ["Fair", "Arbitration", "Unilateral Change", "Content Removal", "Jurisdiction", "Choice of Law", "Limitation of Liability", "Unilateral Termination", "Contract by Using", "Privacy Policy Included"]

    def get_unfair_labels(self):
        return self.labels[1:]

    def labels_to_id(self, exclude_fair=False):
        labels = self.labels[exclude_fair:]
        return {k:v for k,v in zip(labels, range(len(labels)))}
    
    def id_to_labels(self, exclude_fair=False): 
        labels = self.labels[exclude_fair:]
        return {v:k for k,v in zip(labels, range(len(labels)))}

LABELS = Labels()

CHAIN_0_DEF = {
    "a" : "An arbitration clause requires or allows the parties to resolve their disputes through the arbitration, before the case could go to court.",
    "ch" : "A unilateral change clause specifies if and under what conditions the provider can unilaterally change and modify the contract and/or the service.",
    "cr" : "A content removal clause gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content.",
    "j" : "A jurisdiction clause specifies what courts have the competence to adjudicate dispute.",
    "law" : "A choice of law clause specifies what law will govern the contract and be applied in potential disputes.",
    "ltd" : "A limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit or reduce their liability, the duty to compensate damages and/or when contains a blanket phrase like 'to the fullest extent permissible by law'.",
    "ter" : "The unilateral termination clause states that the provider has the right to suspend and/or terminate the service and/or the contract and/or the consumer’s account, due to some reasons, or at any time, for any or no reasons with or without notice. ",
    "use" : "A contract by using clause states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website.",
    "pinc" : "A clause that explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and form part of the terms and it is preceded by a content by using clause to such terms."
}

# CHAIN_1_DEF = {
#     "a" : "A clause is unfair whenever the arbitration is binding and not optional and/or should take place in a country different from the consumer’s place of residence and/or be based not on law but on other arbitration rules and/or arbiter’s discretion.",
#     "ch" : "Such clause is always unfair.",
#     "cr" : "Such clause is always unfair.",
#     "j" : "A clause is unfair whenever it states that judicial proceeding takes a residence away (i.e., in a different city, different country from the consumer place of residence).",
#     "law" : "A clause is unfair whenever it states that the applicable law is different from the law of the consumer’s place of residence.",
#     "ltd" : "Such clause is always unfair, unless it is a force majeure case.",
#     "ter" : "Such clause is always unfair.",
#     "use" : "Such clause is always unfair.",
#     "pinc" : "Such clause is always unfair."
# }

CHAIN_1_DEF = {
    "a" : "A clause is unfair whenever the arbitration is binding and not optional and/or should take place in a country different from the consumer’s place of residence and/or be based not on law but on other arbitration rules and/or arbiter’s discretion.",
    "ch" : "",
    "cr" : "",
    "j" : "A clause is unfair whenever it states that judicial proceeding takes a residence away (i.e., in a different city, different country from the consumer place of residence).",
    "law" : "A clause is unfair whenever it states that the applicable law is different from the law of the consumer’s place of residence.",
    "ltd" : "Such clause is always unfair, unless it is a force majeure case.",
    "ter" : "",
    "use" : "",
    "pinc" : ""
}
