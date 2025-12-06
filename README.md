# HW6 Part 3 - RAG Pipeline

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Set your CMU AI Gateway API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:CMU_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set CMU_API_KEY=your-api-key-here
```

**Alternative:** Use `--api-key` flag when running commands

**Note on Terminals:**
- These commands work in **CMD**, **PowerShell**, and **Git Bash**
- Git Bash users: Use `export CMU_API_KEY="your-key"` instead of `set` or `$env:`

### 3. Prepare Data
Make sure your data is structured as:
```
hw6_project/
├── data/
│   ├── corpus/          # Your 53 documents
│   ├── question.tsv     # 158 questions
│   ├── answer.tsv       # Ground truth answers
│   └── evidence.tsv     # Evidence mapping
├── src/
│   ├── retriever.py
│   ├── generator.py
│   └── rag.py
└── output/
    └── prediction/      # Output will go here
```

## Running the 5 Systems

**Important**: Run all commands from the project root directory (where this README is located).

### System 1: None_gpt4omini (No Retrieval Baseline)
```bash
python src/rag.py --retriever none --generator gpt4omini
```

### System 2: Azure_gpt4omini (API Retriever + API Generator)
```bash
python src/rag.py --retriever azure --generator gpt4omini
```

### System 3: Local_gpt4omini (Open-weight Retriever + API Generator)
```bash
python src/rag.py --retriever local --generator gpt4omini
```

### System 4: Azure_flant5 (API Retriever + Open-weight Generator)
```bash
python src/rag.py --retriever azure --generator flant5
```

### System 5: Local_flant5 (Open-weight Retriever + Open-weight Generator)
```bash
python src/rag.py --retriever local --generator flant5
```

## Output Files

Each system will generate a TSV file in `output/prediction/`:
- `None_gpt4omini.tsv`
- `Azure_gpt4omini.tsv`
- `Local_gpt4omini.tsv`
- `Azure_flant5.tsv`
- `Local_flant5.tsv`

Format: `<prediction>\t<retrieved_docs>`

## Notes

- **Embeddings are cached**: First run will be slower as embeddings are generated
- **Azure retriever**: Uses `text-embedding-3-small` 
- **Local retriever**: Uses `sentence-transformers/all-MiniLM-L6-v2`
- **API generator**: Uses `gpt-4o-mini-2024-07-18`
- **Open-weight generator**: Uses `google/flan-t5-base`

## Troubleshooting

**Import errors:** Make sure all packages are installed
```bash
pip install openai pandas numpy sentence-transformers transformers torch tqdm
```

**API errors:** Verify your API key is set correctly

**Memory issues with FLAN-T5:** The model requires ~1GB RAM. If issues occur, close other applications.

## Command-Line Options

Full usage:
```bash
python src/rag.py --retriever <azure|local|none> \
              --generator <gpt4omini|flant5> \
              [--questions <path>] \
              [--corpus <path>] \
              [--output <path>] \
              [--api-key <key>] \
              [--base-url <url>]
```
