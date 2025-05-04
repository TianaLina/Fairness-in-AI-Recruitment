import pandas as pd
from tqdm import tqdm
import os
import re
import json


def remove_unwanted_columns_recursive(root_folder:str) -> None:
    '''
    The function for cleaning experiments resilts csv-files
    
    Args:
    root_folder(str): path to folder where csv-files are located

    Returns: None
    '''
    additional_cols = ['decision', 'feedback', 'raw_ai_decision']

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing: {file_path}")

                try:
                    df = pd.read_csv(file_path)

                    # Columns containing 'json_extracted'
                    cols_to_drop = [col for col in df.columns if ('json_extracted' in col) or ('Unnamed' in col)]

                    # Add explicitly named columns (if present in df)
                    cols_to_drop += [col for col in additional_cols if col in df.columns]

                    if cols_to_drop:
                        df.drop(columns=cols_to_drop, inplace=True)
                        df.to_csv(file_path, index=False)
                        print(f"Removed columns: {cols_to_drop}")
                    else:
                        print("No matching columns to remove.")

                except Exception as e:
                    return f"Failed to process {file_path}: {e}"



def extract_json(text:str) -> str:
    '''
    Regex to extract json-like parts from raw outputs
    Args:
    text (str): model's output
    Returns:
    str: only json-like part of string or the input string if json was not found
    '''
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text

def parse_decision_feedback(text:str) -> str:
    '''
    Experiment output parser. Looks for decision-feedback pairs in the text input and transforms it into json-like string
    Args:
    text (str): model's output
    Returns:
    str: parsed json-like string or the input string if decision-feedback pair was not found
    '''
    if not isinstance(text, str):
        return ""

    cleaned = text.replace('\\n', '\n')
    cleaned = re.sub(r'print\(.*?\)', '', cleaned, flags=re.DOTALL)

    result = {}

    # 1. Match commented JSON lines like #"decision": "найняти"
    commented_matches = re.findall(r'#"\s*(decision|feedback)"\s*:\s*["\']([^"\']+)["\']', cleaned)
    if commented_matches:
        result.update({k: v for k, v in commented_matches})

    # 2. Match regular key-value with : or =
    matches = re.findall(r'\b(decision|feedback)\s*[:=]\s*["\']([^"\']+)["\']', cleaned)
    if matches:
        result.update({k: v for k, v in matches})

    # 3. Fallback: fix lines like decision: "найняти'
    fallback_matches = re.findall(r'\b(decision|feedback)\s*[:=]\s*["\'](.*?)["\']?', cleaned)
    for key, val in fallback_matches:
        if key not in result and val.strip():
            val = val.strip(' "\'')
            result[key] = val

    return json.dumps(result, ensure_ascii=False) if result else text


def data_clean(df:pd.DataFrame, model_name:str, delimiter:str=None, sep_part:int=-1) -> int:
    '''
    Function that cleans the raw output and extracts decision and feedback provided by the model

    Args:
    df (pd.DataFrame): table with expeiments results
    model_name (str): the model which results should be cleaned
    delimiter (str): delimiter to split raw output, if None, raw output is not splitted. Feasible when model repeats the same json string multiple times
    sep_part (int: 0 or -1): which part (first if 0 and last if -1) of string to use in the cleaning process

    Returns:
    int: number of rows where 'decision' is still blank after cleaning 
    '''

    if delimiter is not None:
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}"].apply(lambda x: re.split(delimiter, x)[sep_part])
    else:
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}"]
    df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("\n```\n", '', regex=True).replace("\'", '"', regex=True).apply(lambda x: re.sub(r'(?<=\w\s|\w{2})"(?=\s\w|\w{2})', "'", x)).apply(lambda x: re.sub(r'(?<=[a-zA-Zа-яА-ЯєЄіІїЇґҐ])"(?=[a-zA-Zа-яА-ЯєЄіІїЇґҐ])', "'", x)).replace('```json\n', '', regex=True)
    df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\'Adtech platform."', "\'Adtech platform.\'", regex=True).replace('\'fixed prices"', "\'fixed prices\'", regex=True).replace('\'STEP Computer Academy"', "\'STEP Computer Academy\'", regex=True).replace('\'ШАГ"', "\'ШАГ\'", regex=True)
    df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].apply(parse_decision_feedback)
    df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].apply(extract_json)
    for i in tqdm(df[(df[f"decision_{model_name}"].isna())].index):
        try:
            if "{" not in df.loc[i, f"raw_output_{model_name}_json_extracted"]:
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = "{" + df.loc[i, f"raw_output_{model_name}_json_extracted"]
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = df.loc[i, f"raw_output_{model_name}_json_extracted"].replace('(', '')
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = df.loc[i, f"raw_output_{model_name}_json_extracted"].replace('[', '')
            if "}" not in df.loc[i, f"raw_output_{model_name}_json_extracted"]:
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = df.loc[i, f"raw_output_{model_name}_json_extracted"] + "}"
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = df.loc[i, f"raw_output_{model_name}_json_extracted"].replace(')', '')
                df.loc[i, f"raw_output_{model_name}_json_extracted"] = df.loc[i, f"raw_output_{model_name}_json_extracted"].replace(']', '')
            output_dict = json.loads(df.loc[i, f"raw_output_{model_name}_json_extracted"])
            output_dict = {key.strip(): value for key, value in output_dict.items()}
            df.loc[i, f"decision_{model_name}"] = output_dict['decision']
            df.loc[i, f"feedback{model_name}"] = output_dict['feedback']
        except:
            continue

    if len(df[df[f"decision_{model_name}"].isna()].index) != 0:
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("『", '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('』', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('/', ' ', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\n\xa0', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\xa0', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('" "', '"', regex=False)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\\\\', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('} {', ',', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('`', '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("\\'", '"', regex=False)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('}{\\n', ',', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('{,', '{', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('ʻ', '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\\n', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('»', '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('«', '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('.}', '"}', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("')", '"}', regex=False)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("‘", '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("’", '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("“", '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("”", '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace("\\n", "", regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('""', '"', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\\r', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\\t', '', regex=True)
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].replace('\*', ' ', regex=True)
    
        
        
        df[f"raw_output_{model_name}_json_extracted"] = df[f"raw_output_{model_name}_json_extracted"].apply(extract_json)
        for i in tqdm(df[df[f"decision_{model_name}"].isna()].index):
            try:
                output_dict = json.loads(df.loc[i, f"raw_output_{model_name}_json_extracted"])
                df.loc[i, f"decision_{model_name}"] = output_dict['decision']
                df.loc[i, f"feedback{model_name}"] = output_dict['feedback']
            except:
                continue
    return len(df[df[f"decision_{model_name}"].isna()].index)

