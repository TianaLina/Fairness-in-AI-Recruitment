import re
import numpy as np
from transformers import AutoTokenizer
from src.constants import PROMPTS
from src.constants import format_instructions
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from collections import defaultdict


def extract_tokens_and_shap(prompt_text:str, shap_values, tokenizer:AutoTokenizer):
    """
    Extracts tokens and SHAP values from the explanation object.
    Returns a list of (token, shap_value) tuples.
    """
    # Get tokenized prompt aligned with SHAP
    tokens = shap_values.data[0]  # List of tokens (already aligned with SHAP)
    values = shap_values.values[0]  # SHAP importance per token

    return list(zip(tokens, values))

def plot_top_k_tokens(tokens_shap, top_k=20):
    """
    Plot top_k tokens with highest absolute SHAP values.
    """
    # Sort by absolute importance
    sorted_tokens = sorted(tokens_shap, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    tokens, values = zip(*sorted_tokens)

    plt.figure(figsize=(10, 4))
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.barh(tokens, values, color=colors)
    plt.xlabel("SHAP Value (Contribution to Output)")
    plt.title(f"Top {top_k} Tokens by SHAP Value")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def highlight_tokens_html(tokens_shap, max_tokens=200):
    """
    Render HTML with tokens color-coded by SHAP value.
    """
    tokens_shap = tokens_shap[:max_tokens]
    html = ""
    for token, val in tokens_shap:
        color = f"rgba(255, 0, 0, {min(abs(val)/max(1, max(abs(v) for _, v in tokens_shap)), 1)})" if val < 0 \
                else f"rgba(0, 128, 0, {min(abs(val)/max(1, max(abs(v) for _, v in tokens_shap)), 1)})"
        html += f"<span style='background-color:{color}; padding:2px; margin:1px; border-radius:3px'>{token}</span> "

    display(HTML(html))



def get_tokens_shap_with_positions(prompt_text, shap_values_df, tokenizer):
    """
    Aligns tokens, shap values, and their character positions.
    
    Inputs:
        - prompt_text: full input prompt (str)
        - shap_values_df: pd.DataFrame with 'shap_value' per token index
        - tokenizer: HuggingFace tokenizer (must support 'return_offsets_mapping')
    
    Returns:
        List of (token_text, shap_value, start_char, end_char)
    """

    # Tokenize with offsets to get (start_char, end_char) for each token
    encoding = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Sanity check
    if len(tokens) != len(shap_values_df):
        raise ValueError(f"Mismatch: {len(tokens)} tokens vs {len(shap_values_df)} SHAP values")

    # Build tokens_shap
    tokens_shap = []
    for token, (start_char, end_char), shap_val in zip(tokens, offsets, shap_values_df['shap_value']):
        tokens_shap.append((token, shap_val, start_char, end_char))

    return tokens_shap

def parse_sections(prompt_text):
    """
    Automatically detect sections using known markers in the prompt text.
    Returns a dict {section_name: (start_idx, end_idx)}
    """
    # Patterns to detect sections
    patterns = {
        "intro": r"You are a smart AI hiring system.*?Job description: ```",
        "job_description": r"Job description: ```(.*?)```",
        "candidate_profile": r"Candidate profile: ```(.*?)```",
        "format_instructions": r"Based on all the information about the candidate.*?```(.*?)```",
    }

    sections = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            start = match.start(1) if match.lastindex else match.start()
            end = match.end(1) if match.lastindex else match.end()
            sections[name] = (start, end)
        else:
            print(f"Warning: Section '{name}' not found!")

    return sections


def assign_token_to_section(token_start, sections):
    """
    Assigns a token to a section based on character start index.
    """
    for section_name, (start, end) in sections.items():
        if start <= token_start < end:
            return section_name
    return None  # If not matching anything


def aggregate_shap_sectionwise(tokens_shap, sections):
    """
    Aggregate shap values per section.
    tokens_shap: list of (token_text, shap_value, token_start, token_end)
    sections: dict of {section_name: (start_idx, end_idx)}
    """
    section_shap_values = defaultdict(list)

    for token, shap_value, token_start, token_end in tokens_shap:
        section = assign_token_to_section(token_start, sections)
        if section is not None:
            section_shap_values[section].append(shap_value)

    section_mean_shap = {section: np.sum(values) for section, values in section_shap_values.items()}
    return section_mean_shap


def plot_section_shap(section_mean_shap):
    """
    Simple horizontal bar plot for section-wise mean SHAP values.
    """
    plt.figure(figsize=(8, 4))
    plt.barh(list(section_mean_shap.keys()), list(section_mean_shap.values()), color='skyblue')
    plt.xlabel("Average SHAP value")
    plt.title("Section-wise SHAP Attribution")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


class ShapValues:
    def __init__(self, data, values):
        self.data = [data]
        self.values = [values]