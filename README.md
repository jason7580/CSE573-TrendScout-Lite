# TrendScout AI: RAG + Knowledge Graph for AI Startup Intelligence

**Group 14 | Project 5 | CSE 573 - Semantic Web Mining**

An AI-powered conversational agent that combines RAG (Retrieval-Augmented Generation) and Knowledge Graph to provide intelligent market intelligence about AI startups.

## 🎯 Project Overview

TrendScout AI monitors the AI startup ecosystem by analyzing LinkedIn posts from leading AI companies. It uses a hybrid approach combining:
- **RAG (ChromaDB + OpenAI embeddings)** for semantic search
- **Knowledge Graph (Neo4j)** for structured relationship queries
- **GPT-4** for answer synthesis

## 📊 Key Results

- **RAG+KG outperforms RAG-only**: 44% win rate vs 14% win rate
- **+1.64 points improvement** on average (out of 15)
- **Best for factual and relationship queries**: +2.8 and +3.3 point improvements

## 🏗️ System Architecture
```
LinkedIn Posts → Entity Extraction (Gemini) → Dual Storage (ChromaDB + Neo4j) → GPT-4 → Web UI
```

## 📁 Project Structure
```
trendscout-ai/
├── scripts/
│   ├── trendscout_app.py              # Main application
│   ├── load_linkedin_posts_kg.py      # Load KG into Neo4j
│   ├── run_evaluation.py              # Evaluation pipeline
│   └── demo_eval.py                   # Single question demo
├── templates/
│   └── index.html                     # Flask web interface
├── data/
│   ├── all_linkedin_posts_combined.json   # 176 LinkedIn posts
│   ├── all_companies_KG_v2.json           # Structured KG data
│   └── Questions/
│       └── eval_questions.json            # 50 evaluation questions
├── eval_results/
│   └── summary_*.json                 # Evaluation results
├── .env.example                       # Environment variables template
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Neo4j Desktop (or Neo4j server)
- OpenAI API key
- Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/trendscout-ai.git
cd trendscout-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Start Neo4j**
- Open Neo4j Desktop
- Create a new database (or use existing)
- Start the database
- Note the URI (default: `neo4j://localhost:7687`)
- Note the password (default: `neo4jneo4j`)

5. **Load Knowledge Graph**
```bash
python scripts/load_linkedin_posts_kg.py
```

6. **Run the application**
```bash
python scripts/trendscout_app.py
```

7. **Open web interface**
```
http://localhost:5001
```

## 🧪 Running Evaluation

To reproduce our evaluation results:
```bash
python scripts/run_evaluation.py \
    --questions data/Questions/eval_questions.json \
    --output ./eval_results
```

To test a single question:
```bash
python scripts/demo_eval.py \
    --question "Which companies partner with government agencies?" \
    --expected "Perplexity (GSA), Anthropic (Maryland, CAISI, AISI)"
```

## 📊 Dataset

- **Source**: LinkedIn posts from 5 AI companies
- **Companies**: Perplexity AI, OpenAI, Anthropic, Mistral AI, DeepSeek
- **Size**: 176 posts (March - November 2025)
- **Knowledge Graph**:
  - 5 Companies
  - 176 Posts
  - 144 Products
  - 63 AI Models
  - 148 Partners

## 🔧 Configuration

Edit `.env` file:
```env
# OpenAI
OPENAI_API_KEY=your-openai-key

# Gemini (for evaluation)
GEMINI_API_KEY=your-gemini-key

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

## 👥 Team Members

- Aditya Pokharna
- Sahil Pawar
- Wei-An Wang
- Yu-Yao Hsieh
- Zih-Jyun Lin

## 📄 License

This project is for academic purposes (CSE 573 - Fall 2025).

## 🙏 Acknowledgments

- Arizona State University
- Professor Hasan Davulcu
- CSE 573 - Semantic Web Mining Course