# UAV Research Copilot

A small Retrieval-Augmented Generation (RAG) project for querying UAV research papers.

## Goal
Load PDFs from `data/papers/`, chunk them, embed them, retrieve relevant chunks, 
and answer questions using an LLM.

## Project Structure

uav-research-copilot/
├── data/
│   └── papers/
├── src/
│   ├── ingest.py
│   ├── rag_pipeline.py
│   ├── prompts.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md

## Quick Start

1. Put UAV research PDFs in `data/papers/`
2. Install dependencies

pip install -r requirements.txt

3. Build the vector index

python src/ingest.py

4. Run the app

streamlit run app.py

## Example Questions

- What acoustic features are used for UAV detection?
- What datasets are used in UAV audio research?
- What are limitations of microphone-based detection?

Author: Roei Yanku
