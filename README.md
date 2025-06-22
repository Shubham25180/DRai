# 🧠 DrAI Medical Tutor

**Your AI-powered companion for NEET-PG preparation**

A comprehensive medical education platform that combines AI-powered doubt clearance, automated notes generation, personalized mock tests, and daily motivation to help medical students excel in their NEET-PG journey.

## 🚀 Features

### 1. ❓ **Doubt Clearance (Chatbot)**
- Ask any medical question and get accurate, student-friendly explanations
- Powered by advanced AI models (BioMistral-7B compatible)
- Comprehensive answers with clinical context

### 2. 📝 **Notes Generator**
- Generate structured, comprehensive notes for any medical topic
- Perfect for quick revision and concept clarification
- NEET-PG focused content

### 3. 📊 **Personalized Mock Tests**
- Topic-specific MCQ practice
- 5 medical specialties covered (Cardiology, Neurology, Respiratory, Gastroenterology, Endocrinology)
- Instant scoring and detailed explanations
- Progress tracking

### 4. 💪 **Daily Motivation Ping**
- Inspirational quotes for medical students
- Continuous motivation service
- Customizable intervals

## 🧪 Step 2: Custom Dataset Preparation

### 📊 **Fine-Tuning Dataset**
The project includes a comprehensive dataset preparation system for fine-tuning AI models on NEET-PG content:

#### **Dataset Files**
```
data/
├── starter_dataset.jsonl      # 🎯 25+ medical Q&A entries
└── dataset_template.csv       # 📋 Google Sheets template
```

#### **Dataset Contents**
- **10 Q&A entries** (doubt clearance)
- **5 MCQ explanations** (mock test generation)  
- **5 Topic summaries** (notes generation)
- **5 Clinical cases** (case-based learning)

#### **Topics Covered**
- Cardiology, Neurology, Respiratory
- Gastroenterology, Endocrinology
- Infectious Diseases, Emergency Medicine
- Nephrology, Surgery, Ophthalmology

#### **Dataset Tools**
1. **`validate_dataset.py`** - Check dataset format and get statistics
2. **`data_preparation.py`** - Convert CSV to JSONL format
3. **`data/dataset_template.csv`** - Template for adding more data

#### **Using the Dataset**
```bash
# Validate your dataset
python validate_dataset.py

# Add more data using the template
# Open data/dataset_template.csv in Google Sheets
# Add entries and convert to JSONL format
```

#### **Dataset Format (JSONL)**
Each entry follows the instruction-tuning format:
```json
{
  "instruction": "Q: What is the treatment for typhoid fever?",
  "input": "",
  "output": "Ceftriaxone or azithromycin for 7-14 days...",
  "topic": "Infectious Diseases",
  "type": "qa"
}
```

#### **Ready for Fine-Tuning**
- ✅ Proper JSONL format
- ✅ NEET-PG focused content
- ✅ Multiple question types
- ✅ Clinical context
- ✅ Compatible with BioMistral-7B fine-tuning

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (for AI model loading)
- Internet connection for model download

### Setup Instructions

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd ai-medical-tutor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the interface**
   - Open your browser and go to: `http://localhost:7860`
   - Or use the provided Gradio share link

## 📁 Project Structure

```
ai-medical-tutor/
├── app.py                  # Main Gradio application
├── utils.py                # Utility functions and prompts
├── mock_questions.json     # MCQ database
├── motivation.py           # Motivation script
├── data_preparation.py     # Dataset preparation tools
├── validate_dataset.py     # Dataset validation script
├── demo.py                 # Feature demonstration
├── data/                   # Dataset directory
│   ├── starter_dataset.jsonl
│   └── dataset_template.csv
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 How to Use

### 1. **Setup (First Time)**
- Go to the "🚀 Setup" tab
- Click "Load AI Model" to initialize the AI system
- Wait for the model to load (may take a few minutes)

### 2. **Doubt Clearance**
- Navigate to "❓ Doubt Clearance" tab
- Type your medical question
- Click "Get Answer" for AI-powered explanation

### 3. **Notes Generation**
- Go to "📝 Notes Generator" tab
- Enter a medical topic (e.g., "Types of Shock")
- Click "Generate Notes" for comprehensive notes

### 4. **Mock Tests**
- Select "📊 Mock Test" tab
- Choose a topic from the dropdown
- Set number of questions (1-10)
- Click "Start Test"
- Answer questions in A,B,C,D format
- Submit for instant results and explanations

### 5. **Motivation**
- Visit "💪 Motivation" tab
- Click "Get Motivation" for inspirational quotes

## 🔧 Advanced Usage

### Running Motivation Service Separately
```bash
python motivation.py
```

### Dataset Management
```bash
# Validate your dataset
python validate_dataset.py

# Use data preparation tools
python data_preparation.py
```

### Customizing MCQs
Edit `mock_questions.json` to add your own questions:
```json
{
  "your_topic": [
    {
      "question": "Your question here?",
      "options": {
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      },
      "correct_answer": "A",
      "explanation": "Explanation here"
    }
  ]
}
```

### Using Different AI Models
Modify the `model_name` parameter in `app.py`:
```python
# For BioMistral-7B (recommended for medical content)
model_name = "mistralai/BioMistral-7B"

# For other models
model_name = "microsoft/DialoGPT-medium"
```

## 🎨 Features in Detail

### AI Model Integration
- **Primary**: BioMistral-7B (medical domain expertise)
- **Fallback**: DialoGPT-medium (general purpose)
- **Optimization**: Quantized loading for memory efficiency

### Mock Test System
- **Topics**: Cardiology, Neurology, Respiratory, Gastroenterology, Endocrinology
- **Questions**: 15+ questions per topic
- **Scoring**: Percentage-based with explanations
- **Randomization**: Questions randomly selected for variety

### Motivation System
- **Quotes**: 10+ medical student-focused quotes
- **Scheduling**: Configurable intervals (30 min to 24 hours)
- **History**: Quote tracking and logging

### Dataset System
- **Format**: JSONL for fine-tuning compatibility
- **Types**: Q&A, MCQ explanations, summaries, clinical cases
- **Validation**: Automatic format checking and statistics
- **Expansion**: Easy template-based data addition

## 🔒 Privacy & Security

- **Local Processing**: All AI processing happens locally
- **No Data Collection**: Your questions and answers stay private
- **Offline Capable**: Works without internet after model download

## 🚀 Performance Tips

1. **Model Loading**: Use quantized models for faster loading
2. **Memory**: Close other applications when loading large models
3. **GPU**: Enable CUDA if available for faster inference
4. **Questions**: Keep questions concise for better responses

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails**
- Check internet connection
- Ensure sufficient RAM (8GB+)
- Try restarting the application

**Mock Test Not Working**
- Verify `mock_questions.json` exists
- Check file permissions
- Restart the application

**Gradio Interface Issues**
- Clear browser cache
- Try different browser
- Check if port 7860 is available

**Dataset Issues**
- Run `python validate_dataset.py` to check format
- Ensure JSONL file is properly formatted
- Check file encoding (should be UTF-8)

### Error Messages

- `"Model not loaded"`: Load the AI model first
- `"Topic not found"`: Check available topics in dropdown
- `"Invalid answer format"`: Use A,B,C,D format for answers
- `"File not found"`: Check if dataset files exist in data/ directory

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add More MCQs**: Expand the question database
2. **Improve Prompts**: Enhance AI response quality
3. **New Features**: Suggest and implement new functionality
4. **Bug Reports**: Report issues and suggest fixes
5. **Dataset Expansion**: Add more medical Q&A entries
6. **Fine-tuning**: Help improve the AI model with custom datasets

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **BioMistral Team**: For the medical AI model
- **Gradio Team**: For the beautiful UI framework
- **Medical Community**: For inspiration and feedback

---

**Built with ❤️ for medical students worldwide**

*Empowering the next generation of healthcare professionals through AI* 