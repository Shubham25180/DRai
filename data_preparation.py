"""
DrAI Medical Tutor - Dataset Preparation Script
Tools for creating and managing fine-tuning datasets for NEET-PG AI tutor
"""

# Import required libraries
import json  # For reading/writing JSON and JSONL files
import pandas as pd  # For handling CSV files
import os  # For file operations
from typing import List, Dict, Any  # For type hints
from datetime import datetime  # For timestamping

class DatasetPreparator:
    """
    Class for preparing, validating, and managing datasets for AI fine-tuning.
    Supports Q&A, MCQ, summary, and clinical case formats.
    """
    def __init__(self):
        # Initialize dataset containers for different entry types
        self.datasets = {
            'qa': [],
            'mcq': [],
            'summary': [],
            'clinical_case': []
        }
    
    def create_qa_entry(self, question: str, answer: str, topic: str = "") -> Dict[str, str]:
        """
        Create a Q&A entry for doubt clearance.
        Returns a dictionary in the required format for fine-tuning.
        """
        return {
            "instruction": f"Q: {question}",
            "input": "",
            "output": answer,
            "topic": topic,
            "type": "qa"
        }
    
    def create_mcq_entry(self, question: str, options: Dict[str, str], 
                        correct_answer: str, explanation: str, topic: str = "") -> Dict[str, Any]:
        """
        Create an MCQ entry with explanation.
        Formats the question, options, and explanation for fine-tuning.
        """
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        instruction = f"Explain this MCQ:\n{question}\n{options_text}"
        output = f"Correct answer: {correct_answer}\n\nExplanation: {explanation}"
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "topic": topic,
            "type": "mcq"
        }
    
    def create_summary_entry(self, topic: str, summary: str) -> Dict[str, str]:
        """
        Create a topic summary entry for notes generation.
        """
        return {
            "instruction": f"Summarize: {topic}",
            "input": "",
            "output": summary,
            "topic": topic,
            "type": "summary"
        }
    
    def create_clinical_case_entry(self, case: str, diagnosis: str, 
                                 explanation: str, topic: str = "") -> Dict[str, str]:
        """
        Create a clinical case entry for fine-tuning.
        """
        return {
            "instruction": f"Clinical Case: {case}",
            "input": "",
            "output": f"Diagnosis: {diagnosis}\n\nExplanation: {explanation}",
            "topic": topic,
            "type": "clinical_case"
        }
    
    def csv_to_jsonl(self, csv_file: str, output_file: str, 
                    instruction_col: str = "instruction",
                    input_col: str = "input", 
                    output_col: str = "output") -> bool:
        """
        Convert a CSV file to JSONL format for fine-tuning.
        Each row in the CSV becomes a JSON object in the output file.
        """
        try:
            df = pd.read_csv(csv_file)
            # Validate required columns
            required_cols = [instruction_col, output_col]
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ Required column '{col}' not found in CSV")
                    return False
            # Convert to JSONL
            with open(output_file, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    entry = {
                        "instruction": str(row[instruction_col]).strip(),
                        "input": str(row.get(input_col, "")).strip(),
                        "output": str(row[output_col]).strip()
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"✅ Successfully converted {csv_file} to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error converting CSV: {e}")
            return False
    
    def validate_dataset(self, jsonl_file: str) -> Dict[str, Any]:
        """
        Validate a JSONL dataset for required fields and content.
        Returns statistics and error details.
        """
        stats = {
            "total_entries": 0,
            "valid_entries": 0,
            "invalid_entries": 0,
            "avg_instruction_length": 0,
            "avg_output_length": 0,
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
            return stats
    
    def create_starter_dataset(self) -> None:
        """
        Create a starter dataset with sample medical Q&A, MCQ, and summary entries.
        Useful for testing and demonstration.
        """
        starter_data = [
            # Q&A Entries
            self.create_qa_entry(
                "What is the first-line treatment for uncomplicated typhoid fever?",
                "Ceftriaxone or azithromycin for 7-14 days. In MDR regions, carbapenems may be used. Supportive care includes hydration and antipyretics.",
                "Infectious Diseases"
            ),
            self.create_qa_entry(
                "Differentiate between nephritic and nephrotic syndrome",
                "Nephritic syndrome: hematuria, hypertension, mild proteinuria (<3.5g/day), oliguria. Nephrotic syndrome: massive proteinuria (>3.5g/day), hypoalbuminemia, edema, hyperlipidemia.",
                "Nephrology"
            ),
            self.create_qa_entry(
                "What are the stages of shock?",
                "1. Compensated: BP normal, tachycardia, cool extremities. 2. Progressive: BP drops, oliguria, confusion. 3. Irreversible: refractory hypotension, multi-organ failure, death.",
                "Emergency Medicine"
            ),
            # MCQ Entries
            self.create_mcq_entry(
                "Which nerve is damaged in Erb's palsy?",
                {"A": "Ulnar nerve", "B": "Median nerve", "C": "Radial nerve", "D": "Upper trunk of brachial plexus"},
                "D",
                "Erb's palsy results from damage to C5-C6 roots (upper trunk of brachial plexus) during birth trauma. Characterized by 'waiter's tip' position.",
                "Neurology"
            ),
            self.create_mcq_entry(
                "What is the most sensitive marker for myocardial injury?",
                {"A": "Creatine kinase", "B": "Troponin I", "C": "Lactate dehydrogenase", "D": "Aspartate aminotransferase"},
                "B",
                "Troponin I is the most sensitive and specific marker for myocardial injury. It rises 3-4 hours after injury and remains elevated for 7-10 days.",
                "Cardiology"
            ),
            # Summary Entries
            self.create_summary_entry(
                "Mechanism of Action of Beta Blockers",
                "Beta blockers competitively inhibit β-adrenergic receptors (β1 and β2). β1 blockade reduces heart rate, myocardial contractility, and renin release. β2 blockade causes bronchoconstriction and vasoconstriction. Used for hypertension, angina, arrhythmias, and heart failure."
            ),
            self.create_summary_entry(
                "Pathophysiology of Diabetes Mellitus Type 2",
                "Type 2 DM results from insulin resistance and relative insulin deficiency. Insulin resistance occurs in liver, muscle, and adipose tissue. Pancreatic β-cells initially compensate with hyperinsulinemia but eventually fail. Risk factors include obesity, family history, and physical inactivity."
            ),
            # Clinical Case Entries
            self.create_clinical_case_entry(
                "21-year-old female presents with fever for 5 days, chills, and rose spots on abdomen. WBC count is low with relative lymphocytosis.",
                "Typhoid fever",
                "Classic presentation of typhoid fever caused by Salmonella typhi. Rose spots are pathognomonic. Confirm with blood culture. Treatment: ceftriaxone or azithromycin.",
                "Infectious Diseases"
            ),
            self.create_clinical_case_entry(
                "45-year-old male with chest pain radiating to left arm, sweating, and nausea. ECG shows ST elevation in leads II, III, aVF.",
                "Inferior wall myocardial infarction",
                "Classic symptoms and ECG findings of inferior wall MI. ST elevation in II, III, aVF indicates right coronary artery occlusion. Requires immediate reperfusion therapy.",
                "Cardiology"
            )
        ]
        
        # Save to JSONL file
        with open('data/starter_dataset.jsonl', 'w', encoding='utf-8') as f:
            for entry in starter_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Created starter dataset with {len(starter_data)} entries")
        print("📁 Saved to: data/starter_dataset.jsonl")
    
    def merge_datasets(self, input_files: List[str], output_file: str) -> bool:
        """Merge multiple JSONL files into one"""
        try:
            merged_data = []
            
            for file in input_files:
                if not os.path.exists(file):
                    print(f"⚠️ File not found: {file}")
                    continue
                
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        merged_data.append(entry)
            
            # Save merged dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in merged_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"✅ Merged {len(merged_data)} entries into {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return False

def main():
    """Main function for dataset preparation"""
    preparator = DatasetPreparator()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("🧠 DrAI Medical Tutor - Dataset Preparation")
    print("="*50)
    print("Choose an option:")
    print("1. Create starter dataset (8 sample entries)")
    print("2. Convert CSV to JSONL")
    print("3. Validate existing dataset")
    print("4. Merge multiple datasets")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        preparator.create_starter_dataset()
        
    elif choice == "2":
        csv_file = input("Enter CSV file path: ").strip()
        output_file = input("Enter output JSONL file path: ").strip()
        preparator.csv_to_jsonl(csv_file, output_file)
        
    elif choice == "3":
        jsonl_file = input("Enter JSONL file path: ").strip()
        stats = preparator.validate_dataset(jsonl_file)
        print(f"\n📊 Dataset Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Valid entries: {stats['valid_entries']}")
        print(f"Invalid entries: {stats['invalid_entries']}")
        print(f"Avg instruction length: {stats['avg_instruction_length']:.1f}")
        print(f"Avg output length: {stats['avg_output_length']:.1f}")
        
        if stats['errors']:
            print(f"\n❌ Errors found:")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
    elif choice == "4":
        files = input("Enter JSONL files to merge (comma-separated): ").strip().split(',')
        output_file = input("Enter output file path: ").strip()
        preparator.merge_datasets(files, output_file)
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 