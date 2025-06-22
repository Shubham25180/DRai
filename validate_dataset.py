"""
Standalone Dataset Validation Script
Run this to check your JSONL dataset format and get statistics
"""

import json
import os

def validate_dataset(jsonl_file):
    """Validate JSONL dataset format and content"""
    if not os.path.exists(jsonl_file):
        print(f"❌ File not found: {jsonl_file}")
        return
    
    stats = {
        "total_entries": 0,
        "valid_entries": 0,
        "invalid_entries": 0,
        "avg_instruction_length": 0,
        "avg_output_length": 0,
        "topics": set(),
        "types": set(),
        "errors": []
    }
    
    instruction_lengths = []
    output_lengths = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats["total_entries"] += 1
                
                try:
                    entry = json.loads(line.strip())
                    
                    # Check required fields
                    required_fields = ["instruction", "output"]
                    for field in required_fields:
                        if field not in entry:
                            stats["errors"].append(f"Line {line_num}: Missing '{field}' field")
                            stats["invalid_entries"] += 1
                            continue
                    
                    # Check field types
                    if not isinstance(entry["instruction"], str) or not isinstance(entry["output"], str):
                        stats["errors"].append(f"Line {line_num}: Fields must be strings")
                        stats["invalid_entries"] += 1
                        continue
                    
                    # Check for empty fields
                    if not entry["instruction"].strip() or not entry["output"].strip():
                        stats["errors"].append(f"Line {line_num}: Empty instruction or output")
                        stats["invalid_entries"] += 1
                        continue
                    
                    # Record lengths
                    instruction_lengths.append(len(entry["instruction"]))
                    output_lengths.append(len(entry["output"]))
                    
                    # Record topics and types
                    if "topic" in entry:
                        stats["topics"].add(entry["topic"])
                    if "type" in entry:
                        stats["types"].add(entry["type"])
                    
                    stats["valid_entries"] += 1
                    
                except json.JSONDecodeError:
                    stats["errors"].append(f"Line {line_num}: Invalid JSON")
                    stats["invalid_entries"] += 1
                    continue
        
        # Calculate averages
        if instruction_lengths:
            stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
        if output_lengths:
            stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        
        return stats
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return None

def main():
    """Main validation function"""
    print("🧠 DrAI Medical Tutor - Dataset Validation")
    print("="*50)
    
    # Check for starter dataset
    starter_file = "data/starter_dataset.jsonl"
    if os.path.exists(starter_file):
        print(f"📁 Found starter dataset: {starter_file}")
        stats = validate_dataset(starter_file)
        
        if stats:
            print(f"\n📊 Dataset Statistics:")
            print(f"Total entries: {stats['total_entries']}")
            print(f"Valid entries: {stats['valid_entries']}")
            print(f"Invalid entries: {stats['invalid_entries']}")
            print(f"Avg instruction length: {stats['avg_instruction_length']:.1f} characters")
            print(f"Avg output length: {stats['avg_output_length']:.1f} characters")
            
            if stats['topics']:
                print(f"\n📚 Topics covered: {', '.join(sorted(stats['topics']))}")
            if stats['types']:
                print(f"🎯 Entry types: {', '.join(sorted(stats['types']))}")
            
            if stats['errors']:
                print(f"\n❌ Errors found ({len(stats['errors'])}):")
                for error in stats['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(stats['errors']) > 5:
                    print(f"  ... and {len(stats['errors']) - 5} more errors")
            
            print(f"\n✅ Validation complete!")
            
            if stats['valid_entries'] > 0:
                print(f"🎉 Your dataset is ready for fine-tuning!")
                print(f"💡 You can now use this with BioMistral or other LLMs")
    
    else:
        print(f"❌ Starter dataset not found at: {starter_file}")
        print("💡 Run the data preparation script first to create it")
    
    # Check for template
    template_file = "data/dataset_template.csv"
    if os.path.exists(template_file):
        print(f"\n📋 Template available: {template_file}")
        print("💡 Open this in Google Sheets or Excel to add more data")

if __name__ == "__main__":
    main() 