"""
String comparison metrics for evaluating similarity between text files.
Includes various algorithms for measuring text similarity.
"""

import os
from difflib import SequenceMatcher
from collections import Counter
import math


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    Returns the minimum number of single-character edits required.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein similarity as a percentage.
    Returns similarity between 0.0 and 100.0.
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100.0
    return ((max_len - distance) / max_len) * 100.0


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaccard similarity coefficient between two strings.
    Based on character-level n-grams.
    """
    set1 = set(s1)
    set2 = set(s2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 100.0
    return (intersection / union) * 100.0


def cosine_similarity(s1: str, s2: str) -> float:
    """
    Calculate cosine similarity between two strings using character counts.
    """
    counter1 = Counter(s1)
    counter2 = Counter(s2)
    
    # Get all unique characters
    all_chars = set(counter1.keys()).union(set(counter2.keys()))
    
    # Create vectors
    vec1 = [counter1.get(char, 0) for char in all_chars]
    vec2 = [counter2.get(char, 0) for char in all_chars]
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return (dot_product / (magnitude1 * magnitude2)) * 100.0


def sequence_matcher_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity using Python's difflib SequenceMatcher.
    """
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio() * 100.0


def hamming_distance(s1: str, s2: str) -> int:
    """
    Calculate Hamming distance between two strings of equal length.
    If lengths differ, returns the difference plus Hamming distance
    of the common length.
    """
    if len(s1) != len(s2):
        # Handle different lengths
        min_len = min(len(s1), len(s2))
        max_len = max(len(s1), len(s2))
        
        # Hamming distance for common part + length difference
        common_distance = sum(c1 != c2 for c1, c2 in
                              zip(s1[:min_len], s2[:min_len]))
        return common_distance + (max_len - min_len)
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hamming_similarity(s1: str, s2: str) -> float:
    """
    Calculate Hamming similarity as a percentage.
    """
    distance = hamming_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100.0
    return ((max_len - distance) / max_len) * 100.0


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro similarity between two strings.
    """
    if len(s1) == 0 and len(s2) == 0:
        return 100.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    match_window = max(len(s1), len(s2)) // 2 - 1
    match_window = max(0, match_window)
    
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len(s1)):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len(s2))
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Find transpositions
    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len(s1) + matches / len(s2) +
            (matches - transpositions / 2) / matches) / 3
    
    return jaro * 100.0


def read_file(file_path: str) -> str:
    """Read the content of a file and return it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def compare_files(file1: str, file2: str) -> dict:
    """
    Compare two files using all available metrics.
    Returns a dictionary with all similarity scores.
    """
    content1 = read_file(file1)
    content2 = read_file(file2)
    
    if not content1 or not content2:
        return None
    
    results = {
        'levenshtein_distance': levenshtein_distance(content1, content2),
        'levenshtein_similarity': levenshtein_similarity(content1, content2),
        'jaccard_similarity': jaccard_similarity(content1, content2),
        'cosine_similarity': cosine_similarity(content1, content2),
        'sequence_matcher_similarity': sequence_matcher_similarity(content1,
                                                                   content2),
        'hamming_distance': hamming_distance(content1, content2),
        'hamming_similarity': hamming_similarity(content1, content2),
        'jaro_similarity': jaro_similarity(content1, content2)
    }
    
    return results


def evaluate_directories(dir1: str, dir2: str, num_files: int = 66) -> dict:
    """
    Evaluate similarity between two directories containing numbered files.
    Returns average scores for all metrics.
    """
    all_results = []
    
    for i in range(num_files):
        file1 = os.path.join(dir1, f"{i}.md")
        file2 = os.path.join(dir2, f"{i}.md")
        
        if os.path.exists(file1) and os.path.exists(file2):
            result = compare_files(file1, file2)
            if result:
                all_results.append(result)
                print(f"File {i}: Levenshtein="
                      f"{result['levenshtein_similarity']:.2f}%, "
                      f"Jaccard={result['jaccard_similarity']:.2f}%, "
                      f"Cosine={result['cosine_similarity']:.2f}%")
    
    if not all_results:
        return None
    
    # Calculate averages
    averages = {}
    for metric in all_results[0].keys():
        if 'distance' not in metric:  # Only average similarity metrics
            total = sum(result[metric] for result in all_results)
            averages[f"avg_{metric}"] = total / len(all_results)
    
    return averages


if __name__ == "__main__":
    # Example usage: compare FT vs groundT directories
    print("Comparing FT vs groundT directories...")
    print("=" * 50)
    
    averages = evaluate_directories("NFT", "groundT", 66)
    if averages:
        print("\n" + "=" * 50)
        print("AVERAGE SIMILARITY SCORES:")
        print("=" * 50)
        for metric, score in averages.items():
            metric_name = metric.replace('avg_', '').replace('_', ' ').title()
            print(f"{metric_name}: {score:.2f}%")
    else:
        print("No files found to compare.")

