import os
import re


def create_directories():
    """
    Create necessary directories for the project
    """
    directories = ['mermaid_outputs', 'FT', 'NFT', 'groundT']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def generateMermaids(image_path, output_path):
    """
    Generate Mermaid diagram from flowchart image using vision model
    """
    from config import model, tokenizer, inference_config
    import torch
    from PIL import Image

    # Load and process image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Load instruction prompt
    with open('instructions.md', 'r') as f:
        instruction_text = f.read()

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction_text}
            ]
        }
    ]

    # Tokenize and generate
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=inference_config.max_new_tokens,
            do_sample=inference_config.do_sample,
            use_cache=inference_config.use_cache
        )

    # Decode output
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Enhanced post-processing for better Mermaid quality
    cleaned_response = post_process_mermaid(response)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(cleaned_response)

    return cleaned_response


def post_process_mermaid(response):
    """
    Enhanced post-processing to fix common Mermaid generation issues
    """
    # Remove any text before the first code block
    if '```' in response:
        start_idx = response.find('```')
        response = response[start_idx:]

    # Ensure proper mermaid code block format
    if not response.startswith('```mermaid'):
        if response.startswith('```'):
            response = '```mermaid\n' + response[3:]
        else:
            response = '```mermaid\n' + response

    # Ensure flowchart TD is present
    lines = response.split('\n')
    if len(lines) > 1 and 'flowchart TD' not in lines[1]:
        lines.insert(1, 'flowchart TD')

    # Fix common syntax issues
    processed_lines = []
    node_connections = {}  # Track connections to avoid duplicates

    for line in lines:
        # Skip empty lines in the middle of flowchart
        if not line.strip() and len(processed_lines) > 1:
            continue

        # Skip the code block markers for processing
        if line.strip() in ['```mermaid', '```', 'flowchart TD']:
            processed_lines.append(line)
            continue

        original_line = line.strip()
        if not original_line:
            continue

        # Fix node shapes based on content
        # Fix decision nodes - should be {text?}
        if '{' in line and '}' in line:
            start = line.find('{')
            end = line.find('}')
            if start != -1 and end != -1:
                question_text = line[start+1:end].strip()
                if not question_text.endswith('?'):
                    question_text += '?'
                line = line[:start] + '{' + question_text + '}' + line[end+1:]

        # Fix input/output nodes - should use [/"text"/]
        if ('input' in line.lower() or 'output' in line.lower() or
            'launched' in line.lower() or 'opened' in line.lower() or
            'saved' in line.lower() or 'muted' in line.lower() or
                'turned off' in line.lower()):
            # Convert to input/output format
            if '[' in line and ']' in line and not line.count('[') > 1:
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    content = line[start+1:end]
                    if not content.startswith('/'):
                        new_content = '[/"' + content + '"/]'
                        line = line[:start] + new_content + line[end+1:]

        # Fix arrow syntax for decision branches
        if '-->' in line and ('yes' in line.lower() or 'no' in line.lower()):
            line = re.sub(r'-->\s*yes\b', '-->|Yes|', line,
                          flags=re.IGNORECASE)
            line = re.sub(r'-->\s*no\b', '-->|No|', line,
                          flags=re.IGNORECASE)

        # Fix broken arrow patterns like "H --> H[Yes]"
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                source = parts[0].strip()
                dest = parts[1].strip()

                # Check for invalid self-reference with different content
                if source and dest:
                    source_node = source.split()[-1] if source else ''
                    dest_node = dest.split()[0] if dest else ''

                    # Fix pattern like "H --> H[Yes]" to "H -->|Yes| I"
                    if (source_node == dest_node and
                            ('[' in dest or '(' in dest or '{' in dest)):
                        # This is likely a malformed decision branch
                        continue  # Skip this malformed line

        # Remove redundant whitespace
        line = re.sub(r'\s+', ' ', line.strip())

        # Track connections to avoid duplicates
        if '-->' in line:
            connection_key = line.split('-->')[0].strip()
            if connection_key not in node_connections:
                node_connections[connection_key] = []
            node_connections[connection_key].append(line)

        if line.strip():
            processed_lines.append(line)

    # Ensure proper closing
    if not any(line.strip() == '```' for line in processed_lines):
        processed_lines.append('```')

    # Join lines and clean up
    result = '\n'.join(processed_lines)

    # Remove any duplicate flowchart TD lines
    result = re.sub(r'(flowchart TD\s*\n)+', 'flowchart TD\n', result)

    # Final validation - ensure we have proper structure
    if '```mermaid' not in result:
        result = '```mermaid\nflowchart TD\n' + result

    return result


def cleanAndProcessFiles(input_dir, output_dir, file_pattern="*.mmd"):
    """
    Clean and process multiple files in a directory
    """
    import glob

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all matching files
    file_pattern_path = os.path.join(input_dir, file_pattern)
    files = glob.glob(file_pattern_path)

    print(f"Found {len(files)} files to process")

    processed_count = 0
    for file_path in files:
        try:
            # Read original file
            with open(file_path, 'r') as f:
                content = f.read()

            # Process content
            cleaned_content = post_process_mermaid(content)

            # Write to output directory
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)

            with open(output_path, 'w') as f:
                f.write(cleaned_content)

            processed_count += 1
            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully processed {processed_count} files")


def process_dataset_images(dataset_path, output_dir, start_idx=0,
                           end_idx=None):
    """
    Process a range of images from dataset to generate Mermaid diagrams
    """
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset(dataset_path)['train']

    if end_idx is None:
        end_idx = len(dataset)

    print(f"Processing images {start_idx} to {end_idx-1}")

    for i in range(start_idx, min(end_idx, len(dataset))):
        try:
            # Get image from dataset
            image = dataset[i]['image']

            # Save temporary image
            temp_image_path = f"temp_image_{i}.png"
            image.save(temp_image_path)

            # Generate Mermaid diagram
            output_path = os.path.join(output_dir, f"flowchart_{i+1:02d}.mmd")
            generateMermaids(temp_image_path, output_path)

            # Clean up temporary file
            os.remove(temp_image_path)

            print(f"Generated: flowchart_{i+1:02d}.mmd")

        except Exception as e:
            print(f"Error processing image {i}: {e}")


def extract_ground_truth(dataset_path, output_dir):
    """
    Extract ground truth Mermaid diagrams from dataset
    """
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset(dataset_path)['train']

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Extracting ground truth for {len(dataset)} items")

    for i, item in enumerate(dataset):
        try:
            # Get Mermaid text
            mermaid_text = item['text']

            # Process and clean the text
            cleaned_text = post_process_mermaid(mermaid_text)

            # Save to file
            output_path = os.path.join(output_dir, f"{i}.md")
            with open(output_path, 'w') as f:
                f.write(cleaned_text)

            print(f"Extracted: {i}.md")

        except Exception as e:
            print(f"Error extracting ground truth {i}: {e}")
