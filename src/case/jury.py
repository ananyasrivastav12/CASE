import os
import sys
import json
import random
from collections import Counter
from typing import List, Dict, Any
from dotenv import load_dotenv

# fix imports to allow running from this directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from models.openrouter_client import OpenRouterClient

# --- legal personas ---
# user-defined personas for evaluating evidence
LEGAL_PERSONAS = {
    "The Strict Textualist": (
        "You are a Strict Textualist following the motto that the text is the law. "
        "Analyze provided evidence only. Do not use outside knowledge. "
        "If evidence does not explicitly state the answer, reject the option. "
        "Catch 'hallucinations' where the model invents rules not found in text."
    ),
    "The Devil's Advocate": (
        "You are a Devil's Advocate. Your goal is to find loopholes in the argument. "
        "Look for exceptions, loopholes, or missing conditions in evidence. "
        "Be highly skeptical. If an answer looks too simple, check for missing conditions."
    ),
    "The Equity Advocate": (
        "You are an Equity Advocate. You view law as a tool for fairness. "
        "In housing/tort cases, consider the vulnerable party for e.g., the tenant. "
        "Interpret ambiguities to prevent unjust outcomes for the vulnerable party."
    ),
    "The Legal Realist": (
        "You are a Legal Realist (Pragmatist). You care about practical consequences. "
        "If literal text leads to absurd results, reject it. "
        "Choose the option that represents a workable, sensible application of rules."
    ),
    "The Precedent Loyalist": (
        "You are a Precedent Loyalist. You care about consistency. "
        "Compare facts in 'Question' strictly against facts in 'Evidence' (Case Law). "
        "If facts don't match, the rule does not apply. Prevent false analogies."
    )
}

class ArbiterDecision(dspy.Signature):
    """
    You are a specialized legal agent with a specific persona.
    Evaluate the multiple choice question using only the retrieved evidence.
    """
    persona_description = dspy.InputField(desc="specific legal philosophy to adopt")
    question = dspy.InputField(desc="legal question to answer")
    options = dspy.InputField(desc="possible answers (A, B, C, D)")
    evidence = dspy.InputField(desc="retrieved legal passages for decision")
    
    reasoning = dspy.OutputField(desc="chain-of-thought. why your persona supports this vote")
    vote = dspy.OutputField(desc="best option letter (A, B, C, or D)")


# --- jury manager ---
class Jury:
    def __init__(self, client: OpenRouterClient = None):
        self.personas = LEGAL_PERSONAS
        
        # config dspy if not already done
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key and not dspy.settings.lm:
            lm = dspy.LM(
                model='openai/meta-llama/llama-3.3-70b-instruct:free',
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                max_tokens=2048
            )
            dspy.settings.configure(lm=lm)

    def synthesize(self, question: str, choices: Dict[str, str], evidence_list: List[Any], num_arbiters: int = 3) -> Dict[str, Any]:
        """
        conducts a trial where random agents review evidence and vote.
        accepts evidence as list of dicts (advocate) or list of json strings (dataset).
        """
        # for now, selects 3 members randomly from the jury personas.
        # formatting inputs
        evidence_text = "\n Evidence - "
        for i, ev in enumerate(evidence_list):
            content = ""

            if isinstance(ev, str):
                try:
                    ev_json = json.loads(ev)
                    content = ev_json.get('contents', str(ev))
                except json.JSONDecodeError:
                    content = ev

            elif isinstance(ev, dict):
                content = ev.get('contents', ev.get('text', str(ev)))
            
            evidence_text += f"[Document {i+1}] {content}\n"

        options_text = "\n".join([f"{k}: {v}" for k, v in choices.items()])

        # select diverse jury
        available_keys = list(self.personas.keys())
        k = min(num_arbiters, len(available_keys))
        selected_names = random.sample(available_keys, k)
        
        print(f"Jury selected: {selected_names}")

        votes = []
        logs = []
        
        # use chain of thought for better reasoning
        predictor = dspy.ChainOfThought(ArbiterDecision)

        # deliberation loop
        for name in selected_names:
            desc = self.personas[name]
            try:
                # agent thinks here
                pred = predictor(
                    persona_description=desc,
                    question=question,
                    options=options_text,
                    evidence=evidence_text
                )
                
                # clean output 
                raw_vote = pred.vote.strip().upper()
                clean_vote = None
                
                # heuristic: find first valid option letter
                for char in raw_vote:
                    if char in choices:
                        clean_vote = char
                        break
                
                if clean_vote:
                    votes.append(clean_vote)
                
                logs.append({
                    "persona": name,
                    "vote": clean_vote,
                    "reasoning": pred.reasoning
                })
                
            except Exception as e:
                print(f"Error with juror {name}: {e}")

        # final verdict (majority vote)
        if not votes:
            return {
                "final_verdict": None,
                "confidence": 0.0,
                "vote_breakdown": {},
                "juror_deliberations": logs
            }

        vote_counts = Counter(votes)
        winner, win_count = vote_counts.most_common(1)[0]
        confidence = win_count / len(votes)

        return {
            "final_verdict": winner,
            "confidence": round(confidence, 2),
            "vote_breakdown": dict(vote_counts),
            "juror_deliberations": logs
        }

if __name__ == "__main__":

    load_dotenv()
    
    print("\nmock trials\n") #update this

    # mock 1
    case_1 = {
        "passage_idx": "mbe_130",
        "question": "Defendant, an avid fan of his home town football team, shot at the leg ofa star playerfor a rival team, intending to injure his leg enough to hospitalize for a few weeks, but not to kill him. The victim died of loss of blood.",
        "choices": {
        "A": "Involuntary manslaughter",
        "B": "Voluntary manslaughter",
        "C": "Murder",
        "D": "None of the above"
        },
        "context_passages": [
        "{\n  \"id\" : \"caselaw_12510790_79\",\n  \"contents\" : \"In total, Scott stabbed the victim five times in the back of the thigh and twice in the calf of his left leg. The victim also had a cut on the right side of his head, a stab wound to his abdomen, a stab wound to his left arm, and a cut on his right forefinger. An autopsy revealed that the stab wounds on the victim's leg severed major arteries in his calf and thigh, which caused extensive bleeding. The State's medical examiner concluded that the victim died as result of blood loss from the stab wounds on his leg.\"\n}",
        "{\n  \"id\" : \"caselaw_12695527_18\",\n  \"contents\" : \"Mr. Murray testified that he was released from the hospital on the date of the accident and that, a few days later, he experienced pain in his neck, shoulders, and back. Mr. Murray went to Ochsner Hospital's emergency room where he was given unspecified medication, placed in an examination room, and then sent home. Prior to being released, Mr. Murray was told to contact Dr. Sarmini to schedule an appointment. After contacting Dr. Sarmini, Mr. Murray underwent an examination wherein he alleges Dr. Sarmini concluded that Mr. Murray's right leg was not getting enough blood flow as a result of the accident. Mr. Murray also stated that he was referred to Ochsner Baptist to see Dr. Walsh who administered epidural shots to his back on four occasions. Mr. Murray testified that the shots were non-effective and that on his next visit to Dr. Sarmini, they discussed the possibility of Mr. Murray having surgery. Mr. Murray stated that he did not have the surgery because of his mother's passing and the need for him to travel back and forth to Monroe because of \\\"family problems.\\\" Mr. Murray also testified that he was unable to go to his therapy sessions because of financial difficulties.\"\n}",
        "{\n  \"id\" : \"caselaw_12508072_3\",\n  \"contents\" : \"see State v. McClain , 154 Conn. App. 281, 283-84, 105 A.3d 924 (2014), aff'd, 324 Conn. 802, 155 A.3d 209 (2017) ; sets forth the following facts: \\\"On July 17, 2010, a group of more than ten people were drinking alcohol in the area known as 'the X,' located behind the Greene Homes Housing Complex in Bridgeport [Greene Homes]. Shortly before 5:22 a.m., the victim, Eldwin Barrios, was sitting on a crate when all of a sudden the [petitioner] and at least two other men jumped on him, and started punching and kicking him. The victim kept asking them why they were hitting him, but no one answered. The [petitioner] then was passed a chrome or silver handgun and he fired one shot, intended for the victim. The bullet, however, struck one of the other men in the back of the leg. The man who had just been shot yelled, 'you shot me, you shot me, why you shot me,' to which the [petitioner] replied, 'my bad.' As this was happening, the victim got up and tried to run away, but the [petitioner] fired several shots at him. Three of the [petitioner's] shots hit the victim-one in the leg, one in the arm, and one in the torso-at which point, the victim fell to the ground and died.\"\n}"
        ]
    }

    # mock 2
    case_2 = {
        "passage_idx": "mbe_533",
        "question": "In 1956, Silo Cement Company constructed a plant for manufacturing ready-mix concrete in Lakeville. At that time Silo was using bagged cement, which caused little or no dust. In 1970, Petrone bought a home approximately 1,800 feet from the Silo plant. One year ago, Silo stopped using bagged cement and began to receive cement in bulk shipments. Since then at least five truckloads of cement have passed Petrone's house daily. Cement blows off the trucks and into Petrone's house. When the cement arrives at the Silo plant, it is blown by forced air from the trucks into the storage bin. As a consequence cement dust fills the air surrounding the plant to a distance of 2,000 feet. Petrone's house is the only residence within 2,000 feet of the plant. If Petrone asserts a claim against Silo based on nuisance, will Petrone prevail?",
        "choices": {
        "A": "Yes, unless using bagged cement would substantially increase Silo's costs.",
        "B": "Yes, if the cement dust interfered unreasonably with the use and enjoyment of Petrone's property.",
        "C": "No, because Silo is not required to change its industrial methods to accommodate the needs of one individual.",
        "D": "No, if Silo's methods are in conformity with those in general use in the industry."
        },
        "context_passages": [
        "{\n  \"id\" : \"caselaw_12574816_12\",\n  \"contents\" : \"¶11 In response, Kilgore explained that the additional height of the silos \\\"doesn't change how fast [the Plant] can run,\\\" \\\"how much material it can produce,\\\" or the \\\"hours of operation.\\\" Even if Kilgore added \\\"eight more 40-foot silos ... [Kilgore] still cannot increase [the number of] trucks under [its] current permit,\\\" which is 150 round-trips per day. Kilgore attributed the \\\"dust issues\\\" to the traffic on \\\"the dirt roads inside the pit,\\\" not the material stored in the silos. And the Division of Air Quality inspected the Plant and informed Kilgore that it was \\\"doing really well.\\\" Kilgore explained that, without approval of the Board, it is permitted to install as many 40-foot silos as it wanted to achieve the same result. Kilgore requested the additional height to reduce the number of silos on the property to achieve the same permitted use.\"\n}",
        "{\n  \"id\" : \"caselaw_12509210_71\",\n  \"contents\" : \"We further find that Amicus' reliance upon our decision in Peterson for affirmance of the Commonwealth Court's decision to be misplaced, as Peterson amply supports our decision reversing that court. Although Amicus is correct that in that case, we held that a zoning ordinance's inclusion of cement manufacturing as a prohibited use in a given district did not extend to also prohibit concrete manufacturing, Amicus ignores the lengthy discussion engaged in by the Peterson Court to differentiate the manufacture of cement from that of concrete. See Peterson , 195 A.2d at 526. Based on a delineation of the different processes for manufacturing cement and concrete, the Court observed that there is \\\"nothing inherently objectionable about the production of concrete, whereas cement manufacture does work discomfort to the inhabitants of the vicinage.\\\" Id. It thus concluded that a cement manufacturing plant is not excluded by the ordinance based on the stated exclusion of concrete manufacturing in the district because it is not the functional equivalent of a concrete manufacturing plant. Id.\"\n}",
        "{\n  \"id\" : \"caselaw_12660364_4\",\n  \"contents\" : \"Cemex owns and operates a cement plant in Odessa, Texas, known as the Cemex-Odessa plant (hereinafter the \\\"Plant\\\"), which is open 24 hours a day, allowing approximately a hundred tractor-trailer trucks to come onto the premises to fill up with cement every day. Union Logistics frequented the plant as an independent contractor hired by Cemex and would send its own employees to fill and deliver loads of cement on behalf of Cemex's customers. Pursuant to its standard procedures, Cemex had trained Union Logistics' owner, Tony Franco, on the Plant's operations, and in turn, Franco was responsible for training his own employees.\"\n}"
        ]
    }

    test_cases = [case_1, case_2]
    jury = Jury()

    for case in test_cases:
        print(f"\n CASE ({case['passage_idx']})")
        print(f"Question: {case['question']}...")
        
        result = jury.synthesize(
            question=case['question'],
            choices=case['choices'],
            evidence_list=case['context_passages'],
            num_arbiters=3
        )
        
        print(f"\nFinal verdict: {result['final_verdict']} (Confidence: {result['confidence']})")
        print(f"Vote tally: {result['vote_breakdown']}")
        
        for log in result['juror_deliberations']:
            print(f"\n[{log['persona']}] voted {log['vote']}")
            print(f"Reasoning: {log['reasoning']}")