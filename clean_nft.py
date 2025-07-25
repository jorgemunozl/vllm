import os
import re


def extract_mermaid_from_file(file_path):
    """Extract clean Mermaid code from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove everything above line 39 first
        lines = content.split('\n')
        if len(lines) > 39:
            content = '\n'.join(lines[39:])
        
        # Method 1: Try to extract from ```mermaid blocks
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        match = re.search(mermaid_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Method 2: Look for flowchart starting lines
        lines = content.split('\n')
        mermaid_start = -1
        
        # Find where flowchart starts
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('flowchart'):
                mermaid_start = i
                break
        
        if mermaid_start >= 0:
            # Extract from flowchart line to end, removing any extra text
            mermaid_lines = []
            for line in lines[mermaid_start:]:
                stripped = line.strip()
                
                # Stop at common end markers
                if (stripped.startswith('```') or 
                    stripped.startswith('assistant') or
                    stripped == '' and len(mermaid_lines) > 5):
                    break
                    
                mermaid_lines.append(line)
            
            return '\n'.join(mermaid_lines).strip()
        
        return None
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def clean_all_output_files():
    """Clean all output files."""
    output_dir = "NFT"
    
    if not os.path.exists(output_dir):
        print("Output directory not found")
        return
    
    md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
    md_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    cleaned_count = 0
    
    for md_file in md_files:
        file_path = os.path.join(output_dir, md_file)
        
        mermaid_code = extract_mermaid_from_file(file_path)
        
        if mermaid_code:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            print(f"✅ Cleaned {md_file}")
            cleaned_count += 1
        else:
            print(f"❌ No Mermaid code found in {md_file}")
    
    print(f"\n🎉 Total cleaned: {cleaned_count} files!")


if __name__ == "__main__":
    clean_all_output_files()
