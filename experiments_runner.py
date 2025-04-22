from vllm import LLM, SamplingParams
import transformers
from tqdm import tqdm
from constants import PROMPTS, format_instructions, format_instructions_uk
import pandas as pd


def run_experiment_vllm_en(df:pd.DataFrame, model:LLM, tokenizer:transformers.AutoTokenizer, model_name="mistralai/Mistral-7B-Instruct-v0.3") -> None:
    """
    Function for running the experiments with English data, explicit injection, and utilizing vLLM library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns: None
    """
    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000)  # Control output length

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            "protected_group": df['protected_group'][i], 
            "protected_attr": df['protected_attr'][i], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions
        }
    
        prompt_text = PROMPTS['baseline_prompt_en'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text

def run_experiment_vllm_uk(df:pd.DataFrame, model:LLM, tokenizer:transformers.AutoTokenizer, model_name="mistralai/Mistral-7B-Instruct-v0.3") -> None:
    """
    Function for running the experiments with Ukrainian data, explicit injection, and utilizing vLLM library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns: None
    """
    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000)  # Control output length

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            "protected_group": df['protected_group'][i], 
            "protected_attr": df['protected_attr'][i], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions_uk
        }
    
        prompt_text = PROMPTS['baseline_prompt_uk'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text



def run_experiment_en(df:pd.DataFrame, model:transformers.AutoModelForCausalLM, tokenizer:transformers.AutoTokenizer, model_name="mistralai/Mistral-7B-Instruct-v0.3") -> None:
    """
    Function for running the experiments with English data, explicit injection, and utilizing transformers library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (transformers.AutoModelForCausalLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns: None
    """
    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    for i in tqdm(df[df[f'decision_{model_name}'].isna()].index):
        prompt_values = {"job_desc": df['Job Description'][i], 
                         "protected_group": df['protected_group'][i], 
                         "protected_attr":df['protected_attr'][i], 
                         "candidate_cv": df['CV'][i],
                        'format_instructions': format_instructions}
    
        prompt_text = PROMPTS['baseline_prompt_en'].format(**prompt_values)

        tokenizer.pad_token = tokenizer.eos_token
 
        input_ids = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1000)

        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        df.loc[i, f'raw_output_{model_name}'] = output

def run_experiment_uk(df:pd.DataFrame, model:transformers.AutoModelForCausalLM, tokenizer:transformers.AutoTokenizer, model_name="mistralai/Mistral-7B-Instruct-v0.3") -> None:
    """
    Function for running the experiments with Ukrainian data, explicit injection, and utilizing transformers library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (transformers.AutoModelForCausalLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns: None
    """
    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {"job_desc": df['Job Description'][i], 
                         "protected_group": df['protected_group'][i], 
                         "protected_attr":df['protected_attr'][i], 
                         "candidate_cv": df['CV'][i],
                        'format_instructions': format_instructions_uk}
    
        prompt_text = PROMPTS['baseline_prompt_uk'].format(**prompt_values)

        tokenizer.pad_token = tokenizer.eos_token
        conversation = [{"role": "user", "content": prompt_text}]
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)
        inputs.to(model.device)
 
        outputs = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=1000, use_tqdm=False)

        output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        df.loc[i, f'raw_output_{model_name}'] = output

def run_experiment_vllm_implicit(df, df_injections, model, model_name="google/gemma-2-27b-it") -> pd.DataFrame:
    """
    Function for running the experiments with English data, implicit injection, and utilizing vLLM library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns:
    pd.DataFrame: input table with experiments results
    """
    
    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000)
    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            'implicit_injection': df_injections['Injection sentence'][df_injections['protected_attr'].str.lower()==df['protected_attr'][i].lower()].values[0], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions
        }
    
        prompt_text = PROMPTS['implicit_prompt_en'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text

    return df

def run_experiment_vllm_implicit_uk(df, df_injections, model, model_name="google/gemma-2-27b-it") -> pd.DataFrame:
    """
    Function for running the experiments with Ukrainian data, implicit injection, and utilizing vLLM library

    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns:
    pd.DataFrame: input table with experiments results
    """

    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000)

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            'implicit_injection': df_injections['Injection sentence'][df_injections['protected_attr'].str.lower()==df['protected_attr'][i].lower()].values[0], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions_uk
        }
    
        prompt_text = PROMPTS['implicit_prompt_uk'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text

    return df

def run_experiment_vllm_implicit_optimized(df, df_injections, model, model_name="google/gemma-2-27b-it") -> pd.DataFrame:
    """
    Function for running the experiments with English data, implicit injection, and utilizing vLLM library.
    Parameters were optimized for model to better follow format instructions and improve the quality of the output.
    
    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns:
    pd.DataFrame: input table with experiments results
    """

    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000, temperature=0.0, top_k=1, top_p=1.0)  # optimized parameters for more stable outputs

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            'implicit_injection': df_injections['Injection sentence'][df_injections['protected_attr'].str.lower()==df['protected_attr'][i].lower()].values[0], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions
        }
    
        prompt_text = PROMPTS['implicit_prompt_en'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text

    return df

def run_experiment_vllm_implicit_uk_optimized(df, df_injections, model, model_name="google/gemma-2-27b-it") -> pd.DataFrame:
    """
    Function for running the experiments with English data, implicit injection, and utilizing vLLM library.
    Parameters were optimized for model to better follow format instructions and improve the quality of the output.
    
    Args:
    df (pd.DataFrame): Table with input data for the experiments
    model (vllm.LLM): model instance that should be used to generate outputs
    tokenizer (transformers.AutoTokenizer): tokenizer instance that should be used for input text tokenization
    model_name (str): model name used in columns names

    Returns:
    pd.DataFrame: input table with experiments results
    """

    if f'decision_{model_name}' not in df.columns:
        df[f'decision_{model_name}'] = None
        df[f'feedback{model_name}'] = None
        df[f'raw_output_{model_name}'] = None
    sampling_params = SamplingParams(max_tokens=1000, temperature=0.0, top_k=1, top_p=1.0)  # optimized parameters for more stable outputs

    for i in tqdm(df[df[f'raw_output_{model_name}'].isna()].index):
        prompt_values = {
            "job_desc": df['Job Description'][i], 
            'implicit_injection': df_injections['Injection sentence'][df_injections['protected_attr'].str.lower()==df['protected_attr'][i].lower()].values[0], 
            "candidate_cv": df['CV'][i],
            'format_instructions': format_instructions_uk
        }
    
        prompt_text = PROMPTS['implicit_prompt_uk'].format(**prompt_values)

        outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)

        output_text = outputs[0].outputs[0].text

        df.loc[i, f'raw_output_{model_name}'] = output_text

    return df


