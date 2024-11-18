ENDPOINTS = {
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    # "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama8": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct"
}

ZERO_PROMPT = """
You are tasked with analyzing clauses from terms of service documents to identify categories of unfairness. Each category is described below:

    -Jurisdiction ⟨j⟩: The jurisdiction clause specifies what courts have the competence to adjudicate disputes. A clause is (potentially) unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence).
    -Choice of Law ⟨law⟩: The choice of law clause specifies what law will govern the contract and be applied in potential disputes. A clause is (potentially) unfair whenever it states that the applicable law is different from the law of the consumer's place of residence.
    -Limitation of Liability ⟨ltd⟩: The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like "to the fullest extent permissible by law." Such a clause is always (potentially) unfair unless it pertains to a force majeure case.
    -Unilateral Change ⟨ch⟩: The unilateral change clause specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service. Such a clause is always (potentially) unfair.
    -Unilateral Termination ⟨ter⟩: The unilateral termination clause states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer’s account for any or no reason, with or without notice. Such a clause is always (potentially) unfair.
    -Contract by Using ⟨use⟩: The contract by using clause states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website. Such a clause is always (potentially) unfair.
    -Content Removal ⟨cr⟩: The content removal clause gives the provider the right to modify, delete, or remove the user’s content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content. Such a clause is always (potentially) unfair.
    -Arbitration ⟨a⟩: The arbitration clause requires or allows the parties to resolve their disputes through arbitration before the case could go to court. A clause is (potentially) unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer’s place of residence, and/or be based not on law but on other arbitration rules or the arbiter’s discretion.
    -Privacy Included ⟨pinc⟩: This category includes clauses that (a) explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or (b) state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a "contract by using" clause. Such clauses are always (potentially) unfair.
    -Fair ⟨fair⟩: No unfairness detected in the clause.

Instructions:
    Examine the clause carefully.
    Return a comma-separated list of tags (j, law, ..., pinc) corresponding to all applicable categories.
    If the clause does not exhibit any unfairness, respond with fair only.
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