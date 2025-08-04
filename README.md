🚀 fine-tuned-llm-tool-calling



A modular AI system for LLM-driven tool calling and natural language reasoning—combining a fine-tuned LLM (for function/tool-call generation) with a second LLM (for humanizing technical responses).

📑 Table of Contents
Overview

Workflow

Core Features

Module Features

finetuning.ipynb

mcp_server.py

api.py

prompter.py

Quick Start

Example Inputs & Outputs

File Overview

License

🧩 Overview
Fine-tuned LLM (optimized for tool-calling) interprets user requests and interacts with an MCP (Model Context Protocol) Server for geospatial or file-based operations.

Second LLM (via LangChain) transforms raw JSON/technical responses into clear, human-friendly language.

Easily extensible to any backend tools or APIs via MCP.

🔄 Workflow
mermaid
Copy
Edit
graph LR
A[User Input] --> B[Fine-tuned LLM<br/>(Tool Call JSON)]
B --> C[MCP Server<br/>(File/Geo Processing)]
C --> D[Raw JSON Output]
D --> E[LangChain LLM<br/>(Verbalization)]
E --> F[Human-Readable Response]
User input → Fine-tuned LLM generates tool-call JSON.

JSON is executed on the MCP Server (e.g., file analysis, cropping, NDVI calculation).

Raw output is converted to human-readable text by a second LLM (LangChain).

Result is delivered as clear, actionable feedback.

✨ Core Features
Out-of-the-box support for GeoTIFF, image analysis, and spatial data tools.

Easily extendable for any MCP-supported function/tool.

Designed for frontend integration as an "agentic" backend.

Scalable, modular, and fully open-source.

🧰 Module Features
finetuning.ipynb
Loads and prepares a Qwen2.5 model using Unsloth.

Converts JSON-based tool-calling datasets into ShareGPT format.

Standardizes and tokenizes data for SFT (Supervised Fine-Tuning).

Trains the model with TRL's SFTTrainer.

Saves and merges LoRA adapters.

Demonstrates how to export and test the model with ShareGPT-style input.

mcp_server.py
Loads as a FastMCP agent for geospatial tool-calling.

Analyzes TIFF/JP2 files, returns bounding box, bands, EPSG, etc.

Crops TIFF/JP2 images using coordinates, saves as PNG.

Computes mean NDVI and NDVI at a specified point (x, y).

Returns digital elevation model (DEM) values.

Cleans up temporary directories on exit.

api.py
REST API for LLM-powered tool-calling with natural language output.

Loads a fine-tuned LLM for parsing prompts & function calls.

Integrates with MCP server to execute backend tools.

Uses LangChain + LLM (e.g. Ollama, phi3) to verbalize responses.

Returns both technical and human-friendly results.

prompter.py
Simple command-line interface for interacting with your LLM+MCP backend.

Sends user prompts to the /generate endpoint.

Automatically parses and prints both human-readable and technical outputs.

⚡ Quick Start
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
# (includes Flask, torch, transformers, langchain, fastmcp, etc.)
Prepare and fine-tune your model in finetuning.ipynb.

Run the backend MCP server:

bash
Copy
Edit
python mcp_server.py
Start the API server:

bash
Copy
Edit
python api.py
(Optional) Launch the CLI prompter:

bash
Copy
Edit
python prompter.py
📝 Example Inputs & Outputs
Tool-calling Dataset Format
json
Copy
Edit
{
  "messages": [
    {"role": "user", "content": "Your instruction..."},
    {"role": "assistant", "content": "TOOL_NEEDED: ...\nPARAMS: {...}"}
  ]
}
API Example Output
json
Copy
Edit
{
  "output": "TOOL_NEEDED: analyze_tiff\nPARAMS: {\"filepath\": \"C:/Users/emre/Desktop/abc.tif\"}",
  "tools": [
    {
      "tool_name": "analyze_tiff",
      "params": {"filepath": "C:/Users/emre/Desktop/abc.tif"},
      "tool_result": {"message": "JP2 analyzed successfully.", ...},
      "human_result": "This file contains a 3-band RGB GeoTIFF..."
    }
  ]
}
📁 File Overview
finetuning.ipynb — Main notebook (all code cells & comments)

toolcalling_dataset.jsonl — Example dataset (/content/toolcalling_dataset.jsonl)

lora_model/, merged_model/ — Output model folders

mcp_server.py — FastMCP agent for geospatial tool endpoints

api.py — REST API for LLM+MCP tool-calling

prompter.py — CLI for prompt-testing

📄 License
MIT License
