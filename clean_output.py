import os
import re


def clean_output_files():
    """Clean all output files by removing everything above line 39 and extracting Mermaid code."""
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        print("Output directory not found")
        return
    
    md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
    md_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    cleaned_count = 0
    
    for md_file in md_files:
        file_path = os.path.join(output_dir, md_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove everything above line 39
            if len(lines) > 39:
                content = ''.join(lines[39:])
            else:
                content = ''.join(lines)
            
            # Extract Mermaid code
            mermaid_pattern = r'```mermaid\n(.*?)\n```'
            match = re.search(mermaid_pattern, content, re.DOTALL)
            
            if match:
                clean_content = match.group(1).strip()
                
                # Write back clean content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                print(f"✅ Cleaned {md_file}")
                cleaned_count += 1
            else:
                print(f"❌ No Mermaid block found in {md_file}")
                
        except Exception as e:
            print(f"❌ Error processing {md_file}: {e}")
    
    print(f"\n🎉 Cleaned {cleaned_count} files successfully!")


if __name__ == "__main__":
    clean_output_files()
