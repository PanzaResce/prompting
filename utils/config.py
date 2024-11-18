ENDPOINTS = {
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama8": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct"
}

# ZERO_PROMPT = """
# Your task is to classify clauses from Terms of Service documents to determine if they are unfair according to the following unfairness categories.

# Categories of Unfair Clauses
#     - Jurisdiction <j>: The jurisdiction clause stipulates what courts will have the competence to
#     adjudicate disputes under the contract. Jurisdiction clauses stating that any judicial proceeding takes a residence away
#     (i.e. in a different city, different country) are unfair.

#     - Choice of Law <law>: The choice of law clause specifies what law will govern the contract,
#     meaning also what law will be applied in potential adjudication of a dispute
#     arising under the contract. In every case where the clause defines the applicable law as the law of
#     the consumer’s country of residence, it is considered as unfair.

#     - Limitation of Liability <ltd>: The limitation of liability clause stipulates that the duty to pay damages
#     is limited or excluded, for certain kind of losses, under certain conditions.
#     Clauses that reduce, limit, or exclude the liability of the service provider are marked as unfair.

#     - Unilateral Change <ch>: The unilateral change clause specifies the conditions under which the
#     service provider could amend and modify the terms of service and/or the service itself. 
#     Such clause was always considered as unfair.

#     - Unilateral Termination <ter>: The unilateral termination clause gives provider the right to suspend
#     and/or terminate the service and/or the contract, and sometimes details the
#     circumstances under which the provider claims to have a right to do so. Unilateral termination clauses 
#     that specify reasons for termination were marked as unfair. 
#     Clauses stipulating that the service provider may suspend or terminate the service at any 
#     time for any or no reasons and/or without notice were marked as unfair.

#     - Contract by Using <use>: The contract by using clause stipulates that the consumer is bound by
#     the terms of use of a specific service, simply by using the service, without even
#     being required to mark that he or she has read and accepted them. These clauses are marked as unfair.

#     - Content Removal <cr>: The content removal gives the provider a right to modify/delete user’s
#     content, including in-app purchases, and sometimes specifies the conditions
#     under which the service provider may do so. Clauses that indicate conditions for content removal were marked as
#     unfair, also clauses stipulating that the service provider may
#     remove content in his full discretion, and/or at any time for any or no reasons and/or without 
#     notice nor possibility to retrieve the content are marked as clearly unfair.

#     - Arbitration <a>: The arbitration clause requires or allows the parties to resolve their dis-
#     putes through an arbitration process, before the case could go to court. Clauses stipulating that the 
#     arbitration should take place in a state other then the state of consumer’s residence and/or be based not on law
#     but on arbiter’s discretion were marked as unfair. Clauses defining arbitration as fully optional would have to be marked as fair.

#     - Privacy included <pinc>: Identify clauses stating that consumers consent to the privacy policy simply by using the service. 
#     Such clauses are considered unfair.

# A clause can be assigned to zero or more unfairness categories. If a clause is unfair respond only with the corresponding tags. If a clause is not unfair, respond only with "<fair>".
# """

ZERO_PROMPT = """
Your task is to classify clauses from Terms of Service documents to determine if they are unfair according to the following unfairness categories.

    - Jurisdiction: The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract.  If a  clause states gives consumers a right to bring disputes in their place of residence than the clause is fair. If a  clause states that any judicial proceeding takes a residence away (i.e. in a different city, different country from the consumer place of residence) than the clause is unfair.
    - Choice of Law: The applicable law  clauses specify what law will govern the contract,  meaning also what law will be applied in potential adjudication of a dispute arising under the contract.  If  the clause states that the applicable law  is the law of the consumer’s country of residence than the clause is fair. If  the clause states that the applicable law  is a law different from the law of the consumer’s country of residence than the clause is unfair. 
    - Limitation of Liability: The limitation of liability clause specifies for what actions/events providers exclude, limit or reduce their liability. If the a clause states that the provider may be liable than the clause is fair. If the clause states  that the provider will never be liable for physical injuries, gross negligence, intentional damages, or for  any action taken by other people, for damages or losses under certain conditions, or  when contains a blanket phrase like “to the fullest extent permissible by law”, or general exclusion or limitation of liability, than the clause is unfair.
    - Unilateral Change: The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clause is always considered as unfair.
    - Unilateral Termination: The unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.If the clause states that the service provider has the right to suspend and/or terminate the service and/or the contract  and either specify reasons for suspension and/or termination or states that the provider has these right at any time for any or no reasons and/or without notice than the clause is unfair. 
    - Contract by Using: The contract by using clause stipulates that the consumer is bound by the terms of use/service of a specific service, simply by using the service, downloading an app or visiting the website, without even being required to mark that he or she has read and accepted them. These clauses are  always unfair.
    - Content Removal: The content removal gives the provider a right to modify/delete user’s content, including in-app purchases. If a clause states  that the provider has the right to modify, delate or remove  the consumer's content,   and either   indicate the conditions or it states that the removal can be done in his full discretion, and/or at any time for any or no reasons and/or without notice, nor possibility to retrieve the content, then the clause is unfair.
    - Arbitration: The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court.  If a clause states that the arbitration is fully optional than the clause is fair. If a clause states that the arbitration is binding, i.e., not optional, before a case can go to court and it should take place in a state other then the state of consumer’s residence and/or  not be based not on law but on arbiter’s discretion, than the clause is unfair. 
    - Privacy Included: The privacy included clauses identifies cases where the scope of the consent granted to the Terms of Service, i.e., to the contract,  incorporate also the consent  to the processing of personal data, as stated in the privacy policy. If a clause state that, simply by using the service and/or downloading an app, and/or visiting a website  the consumer consent to the processing of his or her personal data as stated in the privacy policy, or that the privacy policy is incorporated into and form part of the terms of service and that simply  by using the service, downloading an app or visiting the website the consumer is bound by the terms of use/service, then the clause is unfair.

Classify the given clauses with zero or more categories from the following list [Privacy Included, Arbitration, Content Removeal, Contract by Using, Unilateral Termination, Unilateral Change, Limitation of Liability, Choice of Law, Jurisdiction]
"""

FEW_PROMPT = ZERO_PROMPT + """
EXAMPLES
in the event this agreement to arbitrate does not apply to you or your claim, any judicial proceeding will be brought in the federal or state courts of san francisco county, california.
<j>

this agreement shall be governed in all respects by the laws of the state of new york as they apply to agreements entered into and to be performed entirely within new york between new york residents, without regard to conflict of law provisions.
<law>

we are under no obligation to edit or control user content that you or other users post or publish, and will not be in any way responsible or liable for user content.
<ltd>

We also retain the right to create limits on use and storage at our
sole discretion at any time.
<ch>

in case of a dispute with the member who owns the site, we are allowed to ban this member and remove him/her from the service at our discretion.
<ter>

by using, accessing or otherwise utilizing the ada platform even if you do not set up a user account, you accept, acknowledge and avail yourself to these terms and conditions.
<use>

you understand and agree that mozilla reserves the right, at its discretion, to review, modify, or remove any submission that it deems is objectionable or in violation of these terms.
<cr>

to the fullest extent permitted under applicable law and in the interest of resolving disputes between you and alivecor in the most expedient and cost effective manner, you and alivecor agree that every dispute arising in connection with these terms will be resolved by binding arbitration, unless you are a consumer located in a jurisdiction that prohibits the exclusive use of arbitration for dispute resolution.
<a>

the privacy policy is an integral part of this agreement and is expressly incorporated by reference, and by entering into this agreement you agree to (i) all of the terms of the privacy policy, and (ii) grammarly’s use of data as described in the privacy policy is not an actionable breach of your privacy or publicity rights.
<pinc>

you may be able to upload video, images or sounds.
<fair>
"""

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