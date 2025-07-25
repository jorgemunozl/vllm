def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein distance between two strings."""
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


def read_file(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""


if __name__ == '__main__':
    total_similarity = 0
    files_processed = 0
    
    for i in range(66):
        try:
            content1 = read_file(f"output/{i}.md")
            content2 = read_file(f"groundT/{i}.md")
            
            if content1 and content2:
                distance = levenshtein_distance(content1, content2)
                max_len = max(len(content1), len(content2))
                if max_len > 0:
                    similarity = ((max_len - distance) / max_len) * 100
                else:
                    similarity = 100
                
                total_similarity += similarity
                files_processed += 1
                print(f"File {i}: {similarity:.2f}%")
        except Exception as e:
            print(f"Error with file {i}: {e}")
    
    if files_processed > 0:
        average_similarity = total_similarity / files_processed
        print(f"\nAverage similarity: {average_similarity:.2f}%")
    else:
        print("No files processed successfully.")
