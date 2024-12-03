ENDPOINTS = {
    # "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B",
    # "phi3": "microsoft/Phi-3.5-mini-instruct",  
    # "llama8": "meta-llama/Llama-3.1-8B"
    # "llama8": "meta-llama/Meta-Llama-3-8B-Instruct"
}

ZERO_PROMPT_0 = """
Your task is to classify clauses from Terms of Service documents to determine if they are unfair according to the following unfairness categories.

Categories of Unfair Clauses
    - Jurisdiction <j>: The jurisdiction clause stipulates what courts will have the competence to
    adjudicate disputes under the contract. Jurisdiction clauses stating that any judicial proceeding takes a residence away
    (i.e. in a different city, different country) are unfair.
    - Choice of Law <law>: The choice of law clause specifies what law will govern the contract,
    meaning also what law will be applied in potential adjudication of a dispute
    arising under the contract. In every case where the clause defines the applicable law as the law of
    the consumer’s country of residence, it is considered as unfair.
    - Limitation of Liability <ltd>: The limitation of liability clause stipulates that the duty to pay damages
    is limited or excluded, for certain kind of losses, under certain conditions.
    Clauses that reduce, limit, or exclude the liability of the service provider are marked as unfair.
    - Unilateral Change <ch>: The unilateral change clause specifies the conditions under which the
    service provider could amend and modify the terms of service and/or the service itself. 
    Such clause was always considered as unfair.
    - Unilateral Termination <ter>: The unilateral termination clause gives provider the right to suspend
    and/or terminate the service and/or the contract, and sometimes details the
    circumstances under which the provider claims to have a right to do so. Unilateral termination clauses 
    that specify reasons for termination were marked as unfair. 
    Clauses stipulating that the service provider may suspend or terminate the service at any 
    time for any or no reasons and/or without notice were marked as unfair.
    - Contract by Using <use>: The contract by using clause stipulates that the consumer is bound by
    the terms of use of a specific service, simply by using the service, without even
    being required to mark that he or she has read and accepted them. These clauses are marked as unfair.
    - Content Removal <cr>: The content removal gives the provider a right to modify/delete user’s
    content, including in-app purchases, and sometimes specifies the conditions
    under which the service provider may do so. Clauses that indicate conditions for content removal were marked as
    unfair, also clauses stipulating that the service provider may
    remove content in his full discretion, and/or at any time for any or no reasons and/or without 
    notice nor possibility to retrieve the content are marked as clearly unfair.
    - Arbitration <a>: The arbitration clause requires or allows the parties to resolve their dis-
    putes through an arbitration process, before the case could go to court. Clauses stipulating that the 
    arbitration should take place in a state other then the state of consumer’s residence and/or be based not on law
    but on arbiter’s discretion were marked as unfair. Clauses defining arbitration as fully optional would have to be marked as fair.
    - Privacy included <pinc>: Identify clauses stating that consumers consent to the privacy policy simply by using the service. 
    Such clauses are considered unfair.

A clause can be assigned to zero or more unfairness categories. If a clause is unfair respond only with the corresponding tags. If a clause is not unfair, respond only with "<fair>". Do not provide any additional explanations.
Clause:
"""

ZERO_PROMPT_1 = """
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

# FEW_EXAMPLES_POS = {
#     "a": ["Any and all Claims will be resolved by binding arbitration, rather than in court, except you may assert Claims on an individual basis in small claims court if they qualify.", "If we are not able to resolve your Claims within 60 days, you may seek relief through arbitration or in small claims court, as set forth below."],
#     "cr": ["Although we have no obligation to screen, edit, or monitor Your Content, we may, in our sole discretion, delete or remove Your Content at any time and for any reason, including for a violation of these Terms, a violation of our Content Policy, or if you otherwise create liability for us.", "We have the right to remove any posting you make on our site if, in our opinion, your post does not comply with the content standards set out in our Acceptable Use Policy."],
#     "ch": ["If we make changes, we will post the amended Terms to our Services and update the Effective Date above.", "We may also, at our sole discretion, limit access to the Services and/or terminate the accounts of any users who infringe any intellectual property rights of others, whether or not there is any repeat infringement."],
#     "j": ["any judicial proceeding will be brought in the federal or state courts of San Francisco county, California.", "Except as otherwise set forth in these Terms, these Terms shall be exclusively governed by and construed in accordance with the laws of The Netherlands, excluding its rules on conflicts of laws."],
#     "law": ["If you are from outside of mainland China, The Terms shall be governed by the laws of Hong Kong without regard to its conflict of law provisions.", "You irrevocably agree that the courts of England have exclusive jurisdiction to settle any dispute or claim that arises out of or in connection with the Terms or their subject matter or formation (including non-contractual disputes or claims)."],
#     "ltd": ["If we fail to comply with these Terms, we will be liable to you only for the purchase price of the Products in question.", "We will not be liable, directly or indirectly, for any damage or loss caused or alleged to be caused by or in connection with your use of or reliance on any such content, goods or services available on or through any third party websites, content or mobile application."],
#     "ter": ["We reserve the right to delete or disable content alleged to be infringing and terminate accounts of repeat infringers.", "We may terminate your Account(s) if we learn, or in good faith believe, that you are a registered sex offender, that accessing the Service may violate a condition of parole or probation, that you have engaged in or attempted to engage in conduct with minors on the Service that violates this Agreement, or that you for any other reason may pose what we deem to be an unacceptable risk to the Service community."],
#     "use": ["When you use our Services, in addition to enjoying a world of good vibes, you also agree to the Terms and they affect your rights and obligations.", "You agree to our Terms of Service (“Terms”) by installing, accessing, or using our apps, services, features, software, or website (together, “Services”)."],
#     "pinc": ["Any use of the services implies unreserved approval of these terms and Ubisoft's privacy policy.", "Please see our Privacy which forms part of these terms and conditions."]
# }

# FEW_EXAMPLES_HALF = {
#     "a": ["Any and all Claims will be resolved by binding arbitration, rather than in court, except you may assert Claims on an individual basis in small claims court if they qualify.", 
#           "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "cr": ["Although we have no obligation to screen, edit, or monitor Your Content, we may, in our sole discretion, delete or remove Your Content at any time and for any reason, including for a violation of these Terms, a violation of our Content Policy, or if you otherwise create liability for us.", 
#            "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "ch": ["If we make changes, we will post the amended Terms to our Services and update the Effective Date above.", 
#            "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "j": ["any judicial proceeding will be brought in the federal or state courts of San Francisco county, California.", 
#           "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "law": ["If you are from outside of mainland China, The Terms shall be governed by the laws of Hong Kong without regard to its conflict of law provisions.", 
#             "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "ltd": ["If we fail to comply with these Terms, we will be liable to you only for the purchase price of the Products in question.", 
#             "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "ter": ["We reserve the right to delete or disable content alleged to be infringing and terminate accounts of repeat infringers.", 
#             "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "use": ["When you use our Services, in addition to enjoying a world of good vibes, you also agree to the Terms and they affect your rights and obligations.", 
#             "we are not a health care or medical device provider, nor should our products be considered medical advice."],
#     "pinc": ["Any use of the services implies unreserved approval of these terms and Ubisoft's privacy policy.", 
#              "we are not a health care or medical device provider, nor should our products be considered medical advice."]
# }

MULTI_PROMPT = """Evaluate the following clause from a terms of service document based on the provided definition of unfairness:  
Definition:  
{cat_descr}  

If the clause is unfair according to the definition respond with "y" otherwise respond with "n". Respond with "n" also if the clause is unrelated to the definition. Do not include any additional text or explanation.
Clause: {clause}
"""

MULTI_PROMPT_FEW = """Evaluate the following clause from a terms of service document based on the provided definition of unfairness:  
Definition:  
{cat_descr}  

If the clause is unfair according to the definition respond with "y" otherwise respond with "n". Respond with "n" also if the clause is unrelated to the definition. Do not include any additional text or explanation.
Examples:
{examples}

Test Input:
Clause: {clause}
Response:"""

LABEL_TO_ID = {
    "fair": 0,
    "a": 1,
    "ch": 2,
    "cr": 3,
    "j": 4,
    "law": 5,
    "ltd": 6,
    "ter": 7,
    "use": 8,
    "pinc": 9
}
ID_TO_LABEL = {v:k for k, v in LABEL_TO_ID.items()}
