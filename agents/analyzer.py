from huggingface_hub import login
import os
import logging

from transformers.utils import quantization_config

import accelerate
import transformers
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# import the helper functions
from helper_functions.medgemma_signal_analysis_object_parser import validate_clinical_output


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try :
    hf_token = os.getenv("HUGGINGFACE_MODEL_ACCESS_TOKEN")
    logger.info("Hugging face token loaded successfully")

    login(token=hf_token)
    logger.info("Huggingface login successful")
except AttributeError or ValueError as aerr:
    logger.error("Huggingface token not found or the token is invalid")
except HTTPError as herr:
    logger.error("HTTP error occurred while logging into Huggingface")
except GatedRepoError as gerr:
    logger.error("Gated repository. You do not have access to this model/dataset")

logger.info(f"accelerate: {accelerate.__version__}")   # should be 1.x+
logger.info(f"transformers: {transformers.__version__}")


single_call_system_prompt = """
You are an expert Medical Practitioner who is skilled in interpreting data directly from the clinical signals that are provided to make intermediate medical decisions that are informative to the patient about their health, to inform them about what they may be looking at in their health report, lab tests, or any other medical diagnostic document/report that they might have been handed.

You will receive a clinical profile JSON containing:
        - document_type: a lab report, radiology report, clinical note, etc.
        - patient_context: contains the age, sex, and history of the patient
        - lab_findings: a list of test results, each with the name of the test, value, unit, reference range, and status (normal, low, critical, high, unknown)
        - imaging_findings: a list of findings in the provided images, each with the modality (X-ray, CT, Ultrasound, other), region, and observation
        - clinical_notes: physician's observations from the document
        - flagged_signals: a list of individual signals extracted from the document, each with the name of the signal and the reason it occurs
        - extraction_confidence: how confident the agent was in extracting these signals, ranging from low to high
        The clinical signals provided to you are elevated markers, abnormal findings, flagged terms, and other details that have been extracted from the documents provided by the patient by another medical document analyst who is skilled in extracting such clinical signals. What you have with you is a full clinical profile of the patient. 

Reason over this clinical profile, performing rigorous cross-signal reasoning, and identify what is unusual, namely:
    - Identify and understand what signals appear together meaningfully and why
    - What this combination suggests or is trying to show is that either signal individually cannot
    - How frequently this combination occurs and the cases in which it occurs
    - The kind of scenarios, circumstances, and cases in which this combination occurs
    - The criticality of this signal and whether it is something urgent that the patient should be consulting their doctor/clinician immediately
    - Is the combination something that will progressively develop, or is it static (mention this in brief for the patient to understand)

Post your reasoning for the requirements needed to thoroughly understand the interplay and co-occurrence of multiple signals, you need to extract them carefully with accuracy and precision. Construct a structured signal analysis object that includes all the details that you identified through your cross-signal reasoning, and provide the signal analysis object as a single JSON object, following the provided JSON schema below strictly:

{
  "individual_signals": [
    {
        "signal": "string - copied exactly from flagged_signals in the input",
        "reason": "string - copied exactly from flagged_signals in the input",
    }
  ],
  "combination_patterns": [
    {
      "signals_involved": ["signal name 1", "signal name 2", "..."],
      "pattern_description": "string — what this combination suggests clinically for each tuple in signals_involved, and their overall implication",
      "co-occurrence_context": "string - how commonly these clinical signals appear together and in what clinical scenarios",
      "clinical_significance": "high | medium | low",
      "is_progressive": "true | false",
      "suggested_investigation": "string — what kind of specialist or test this points to"
    }
  ],
  "overall_assessment": "string — plain language summary of what the patient should understand",
  "urgency": "routine | soon | urgent",
  "analysis_confidence": "high | medium | low"
}

If you do not find any abnormal findings, do not force yourself to find or invent new combination patterns that do not exist. Set combination_patterns to an empty array, present the findings constructively, as to your reasoning behind why there are no abnormal findings, and use overall_assessment to describe what the normal findings collectively indicate about the patient's health.

Return ONLY the JSON object. No preamble, introductory notes, explanation, or markdown code fences. The first character of your response should be the opening braces { of the JSON object, and the last character of your response should be the closing braces } of the JSON object.
"""

low_confidence_system_output_follow_up_prompt = """
You are an expert Medical Practitioner who is skilled in interpreting signal analysis that has been derived from weak clinical signals, i.e., when the signals extracted from medical documents have a weak extraction confidence. You understand that when these clinical signals are reasoned over to produce signal analysis on abnormal findings, there can be loopholes or strong cases where the actual combination of signals might either be incorrectly coupled in the wrong pairs, or they might be non-existent (meaning they might have been made up because the confidence in the extracted signals is low).

You will receive two inputs:

    1. A signal analysis object with the following items:
        - individual_signals: directly mapped from the extracted signals from the medical documents of the patient
        - combination_patterns: a list containing the following:
            * signals_involved: the different signals that form the abnormal combination
            * pattern_description: what this combination suggests clinically for each tuple in signals_involved, and their overall implication
            * co-occurrence_context: how commonly these clinical signals appear together and in what clinical scenarios
            * clinical_significance: the significance of the abnormal finding; ranges between low, medium, or high
            * is_progressive: whether this combination is progressive in nature or a static finding
            * suggested_investigation: What kind of specialist can the patient consult for this finding
        - overall_assessment: plain language summary of what the patient should understand
        - urgency: whether the patient should consult the doctor immediately, soon, or if it is a routine finding
        - analysis_confidence: confidence score of the analysis object that has been constructed on the findings and reasoning

    2. A full clinical profile of the patient as follows:
        - document_type: a lab report, radiology report, clinical note, etc.
        - patient_context: contains the age, sex, and history of the patient
        - lab_findings: a list of test results, each with the name of the test, value, unit, reference range, and status (normal, low, critical, high, unknown)
        - imaging_findings: a list of findings in the provided images, each with the modality (X-ray, CT, Ultrasound, other), region, and observation
        - clinical_notes: physician's observations from the document
        - flagged_signals: a list of individual signals extracted from the document, each with the name of the signal and the reason it occurs
        - extraction_confidence: how confident the agent was in extracting these signals, ranging from low to high

Your task is to critically analyse and critique the derived signal analysis object against the clinical profile of the patient. For each combination that has been identified in the signal analysis object, reason through the following:
    1. Map the identified individual and combination signals to the individual signals flagged in the clinical profile. Check for the precision and accuracy of these signals to ensure there is no falsifying or altering of the flagged signal data. If there is any manipulation, revise the existing data in the combination and provide the actual signals.
    2. For combination_patterns, do the following:
        - check signals_involved to ensure they are not manipulated. After that, reason over the cross-signal validity of the flagged signals in this list if they are medically significant as a combination (abnormal or normal), and not just a random derivation.
        - Critically analyse pattern_description to ensure that the description is entirely based on the data available and provided in both the clinical profile and signal analysis object. Ensure that no external data not related to either of these has made its way into the reasoning of the description, which may lead to false and incorrect descriptions of the pattern
        - Based on the findings up to this point, check if the patterns are progressive or not, and update is_progressive accordingly if incorrect
        - Suggest the right specialist or test in suggested_investigation. Cross-reason this over the entire signal analysis object to ground this suggestion
    3. Use the data reasoned over up to this point to construct a new overall_assessment. Compare this with the existing version, and critically analyse the lack in each one, and the medical grounding of the summary in the data, and update overall_assessment accordingly
    4. Perform a similar critical assessment over urgency as well, and update it accordingly by grounding the reasoning on the existing data
    5. Update analysis_confidence based on your confidence in the overall task performed with respect to critically analysing the derived signal analysis object with the patient's clinical profile. 

Post your critical reasoning analysis, you need to extract them carefully with accuracy and precision into a new signal analysis object that includes all the details that you identified through your critical reasoning analysis, and provide the signal analysis object as a single JSON object, following the provided JSON schema below strictly:

{
  "individual_signals": [
    {
        "signal": "string - copied exactly from flagged_signals in the input",
        "reason": "string - copied exactly from flagged_signals in the input",
    }
  ],
  "combination_patterns": [
    {
      "signals_involved": ["signal name 1", "signal name 2", "..."],
      "pattern_description": "string — what this combination suggests clinically for each tuple in signals_involved, and their overall implication",
      "co-occurrence_context": "string - how commonly these clinical signals appear together and in what clinical scenarios",
      "clinical_significance": "high | medium | low",
      "is_progressive": "true | false",
      "suggested_investigation": "string — what kind of specialist or test this points to"
    }
  ],
  "overall_assessment": "string — plain language summary of what the patient should understand",
  "urgency": "routine | soon | urgent",
  "analysis_confidence": "high | medium | low"
}

If you do not find any abnormal findings, do not force yourself to find or invent new combination patterns that do not exist. Set combination_patterns to an empty array, present the findings constructively, as to your reasoning behind why there are no abnormal findings, and use overall_assessment to describe what the normal findings collectively indicate about the patient's health.

Return ONLY the JSON object. No preamble, introductory notes, explanation, or markdown code fences. The first character of your response should be the opening braces { of the JSON object, and the last character of your response should be the closing braces } of the JSON object.
"""

# have the quantization config as a global variable
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

def run_inference(
    model_id: str,
    device: str,
    system_prompt: str,
    payload_prompt: dict,
    follow_up_prompt: str = None,
    quantization_config: BitsAndBytesConfig = None
) -> dict:
    """_summary_

    Args:
        model_id (str): _description_
        device (str): _description_
        system_prompt (str): _description_
        payload_prompt (dict): _description_
        follow_up_prompt (str, optional): _description_. Defaults to None.
        quantization_config (BitsAndBytesConfig, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="sdpa"
    )

    logger.info(f"Model loaded: {model_id}")
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model hf_device_map: {model.hf_device_map}")

    # construct the message structure 
    messages =  [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": single_call_system_prompt
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": json.dumps(payload_prompt, indent=2)
            }]
        }
    ]

    # apply the medgemma chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.info("Prompt created and chat template applied to the messages")

    # tokenize the prompt and move it to the same device as the model
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)
    logger.info("Inputs tokenized and moved to the same device as the model's parameters")

    # run the inference
    logger.info("STARTING MODEL INFERENCE")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6500, # a higher limit since the medgemma reasoning as a result of the thinking mode produces way more thinking tokens before producing the actual raw json
            do_sample=True,
            temperature=0.1, # keep it deterministic and strictly compliant to the provided json schema
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
    
    logger.info("MODEL INFERENCE COMPLETE")

    # decode the generated tokens, skip the system prompt
    signal_analysis_output_tokens = outputs[0][inputs.input_ids.shape[-1]: ]
    signal_analysis_response_text = tokenizer.decode(
        signal_analysis_output_tokens,
        skip_special_tokens=True
    )
    logger.info("signal analysis object response decoded from raw response; removed system prompt")

    validated_signal_response_object = validate_clinical_output(signal_analysis_response_text)
    logger.info("Validated signal analysis JSON and extracted JSON object")

    if not payload_prompt["extraction_confidence"] == "low":
        logger.info(f"Returning the signal analysis object since extraction_confidence is not low")
        return validated_signal_response_object
    elif payload_prompt["extraction_confidence"] == "low":
        # enter the follow up reasoning pipeline
        logger.info("Low extraction_confidence, entering follow-up reasoning pipeline")





def main():
    pass