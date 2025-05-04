import shap
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from src.constants import PROMPTS
from src.constants import format_instructions

def llm_predict(prompts:list[str], model:LLM) -> list[str]:
    """Function to pass multiple prompts to the model and return text outputs."""
    sampling_params = SamplingParams(max_tokens=1000)  
    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def scoring_function(prompts:list[str]) -> np.array:
    """Converts LLM outputs into numerical scores for SHAP."""
    responses = llm_predict(prompts)
    return np.array([len(response) for response in responses]) 

def run_experiment_with_shap(df:pd.DataFrame, model:LLM, tokenizer:AutoTokenizer, model_name="google/gemma-2-27b-it") -> pd.DataFrame:
    """Function for running the experiments with explainability framework implemented (SHAP Explainer)"""

    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
        df[f'shap_values_{model_name}'] = None  # Store SHAP values

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            "protected_group": df['protected_group'][i], 
            "protected_attr": df['protected_attr'][i], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions
        }

        prompt_text = PROMPTS['baseline_prompt_en'].format(**prompt_values)

        output_text = llm_predict([prompt_text], model=model)[0]
        df.loc[i, f'raw_output_{model_name}'] = output_text

        masker = shap.maskers.Text(tokenizer)  
        explainer = shap.Explainer(scoring_function, masker)

        shap_values = explainer([prompt_text])  

        df.loc[i, f'shap_values_{model_name}'] = str(shap_values.values.tolist())

    return df
