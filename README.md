# DeepLaw: Legal Document RAG System with Multi-Agent Architecture

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-00A67E.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated legal document analysis system featuring horizontal multi-agent architecture with Retrieval-Augmented Generation (RAG), multi-judge evaluation, and comprehensive performance metrics.

## ✨ Features

### 🤖 **Horizontal Multi-Agent System**
- **PDF Processing Agent**: Document ingestion, vectorization, and storage
- **Data Retrieval Agent**: Context retrieval with confidence scoring
- **Legal Analyzer Agent**: Specialized legal reasoning and analysis
- **Response Generation Agent**: User-facing response creation
- **Quality Assurance Agent**: Multi-judge evaluation and validation
- **Agent Manager**: Orchestrates inter-agent communication and workflow

### 📊 **Comprehensive Evaluation System**
- **Multi-Judge Parallel Evaluation**: Multiple LLM judges evaluate each response
- **Quantitative Metrics**: Accuracy, latency, memory usage, and error tracking
- **Quality Metrics**: Faithfulness, groundedness, relevance, completeness, fluency
- **Interactive Analytics**: Real-time charts and performance visualization

### 🎯 **Legal-Specific Capabilities**
- Legal document parsing and analysis
- Context-aware legal reasoning
- Confidence-based agent triggering
- Specialized legal prompt engineering


## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deeplaw-rag.git
cd deeplaw-rag
```
2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Pull Ollama models (choose at least one)**
```bash
ollama pull llama3.2
ollama pull mistral
```
5. **Run the application**
```bash
streamlit run main.py
```