from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import xml.etree.ElementTree as ET
import re

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, "r", encoding="utf-8") as f:
        xml_content = f.read().replace("&", "&amp;")  

    root = ET.fromstring(xml_content)
    pattern = r"(\d+)-(\d+)"

    sentence_pairs = []
    aligments = []

    for sentence in root.findall('s'):
        pair = SentencePair(sentence.find('english').text.split(), sentence.find('czech').text.split())
        sentence_pairs.append(pair)
        
        sure_alignments = sentence.find('sure').text
        possible_alignments = sentence.find('possible').text

        if sure_alignments is None:
            sure = []
        else:
            sure = [(int(a), int(b)) for a, b in re.findall(pattern, sure_alignments)]

        if possible_alignments is None:
            possible = []
        else:
            possible = [(int(a), int(b)) for a,b in re.findall(pattern, possible_alignments)]
        
        aligment = LabeledAlignment(sure, possible)
        aligments.append(aligment)

    return sentence_pairs, aligments

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    source_counter = Counter()
    target_counter = Counter()

    for sentence_pair in sentence_pairs:
        source_counter.update(sentence_pair.source)
        target_counter.update(sentence_pair.target)
    
    if freq_cutoff is None:
        source_dict = {token: idx for idx, token in enumerate(source_counter.keys())}
        target_dict = {token: idx for idx, token in enumerate(target_counter.keys())}
        return source_dict, target_dict

    source_dict = {token: idx for idx, (token, _) in enumerate(source_counter.most_common(freq_cutoff))}
    target_dict = {token: idx for idx, (token, _) in enumerate(target_counter.most_common(freq_cutoff))}
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for sentence_pair in sentence_pairs:
        if any([token not in source_dict for token in sentence_pair.source]):
            continue
        if any([token not in target_dict for token in sentence_pair.target]):
            continue
        
        tokenized_source = np.array([source_dict[token] for token in sentence_pair.source])
        tokenized_target = np.array([target_dict[token] for token in sentence_pair.target])
        tokenized_sentence_pairs.append(TokenizedSentencePair(tokenized_source, tokenized_target))
    return tokenized_sentence_pairs
    
