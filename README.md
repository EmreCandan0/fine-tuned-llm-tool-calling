# fine-tuned-llm-tool-calling
A fine-tuned LLM optimized for tool-calling from MCP Server and converted to natural language by another LLM.


---------------
A modular AI system that combines:

A fine-tuned LLM (optimized for tool-calling) to interpret user requests and interact with a Model Context Protocol (MCP) Server for geospatial or file-based operations.

A second LLM (via LangChain) that converts raw JSON/technical responses from the tool-calling agent into clear, human-friendly natural language.

Easily extensible for any backend tools or APIs via MCP.
-----------------
Workflow:

User input → Fine-tuned LLM generates tool-call JSON.

JSON is executed on the MCP Server (e.g., file analysis, cropping, NDVI calculation).

Raw output is converted to human-readable text by a second LLM (LangChain).

Result is delivered as clear, actionable feedback to the end user.
------------------
Features:

Out-of-the-box support for GeoTIFF, image analysis, and spatial data tools.

Easily extendable for any function/tool supported by MCP.

Designed for frontend integration as an "agentic" backend.

Scalable, modular, and fully open-source.


-------------------

finetuning.ipynb Features
Loads and prepares a Qwen2.5 model using Unsloth.

Converts JSON-based tool-calling datasets into ShareGPT format.

Standardizes and tokenizes data for SFT (Supervised Fine-Tuning).

Trains the model with TRL's SFTTrainer.

Saves and merges LoRA adapters.

Demonstrates how to export and test the model with ShareGPT-style input.

Example Usage
Run the notebook in Google Colab:

Or view the notebook source.

File Overview
finetuning.ipynb — Main notebook with all code cells and comments

toolcalling_dataset.jsonl — Example dataset (expected path in notebook: /content/toolcalling_dataset.jsonl)

lora_model/, merged_model/ — Output model folders (produced after training)

Quick Start
Upload your dataset:
Format:
{
  "messages": [
    {"role": "user", "content": "Your instruction..."},
    {"role": "assistant", "content": "TOOL_NEEDED: ...\nPARAMS: {...}"}
  ]
}
Edit the notebook if needed (for paths or dataset).

Run all cells in Google Colab or locally (if you have GPU support).

Exported models are saved as .zip files and can be found in your Google Drive.

Example Output
TOOL_NEEDED: analyze_tiff
PARAMS: {"filepath": "C:/Users/emre/Desktop/abc.tif"}<|im_end|>


-----------
mcp_server.py Features
Loads as a FastMCP agent for serving geospatial tool-calling endpoints.

Analyzes TIFF/JP2 files and returns spatial metadata (bounding box, bands, EPSG, etc).

Crops TIFF/JP2 images using bounding box coordinates and saves as PNG.

Computes mean NDVI and NDVI at a specified point (x, y) using red/NIR bands.

Returns digital elevation model (DEM) value at any given point.

Cleans up temporary directories on exit.

-----------
prompter.py Features
Features

Simple command-line interface for interacting with your LLM+MCP backend.

Sends user prompts to the /generate endpoint.

Automatically parses and prints both:

Human-readable results (if available).

Technical tool outputs (as fallback).


------------
api.py Features

REST API for LLM-powered tool-calling with automatic natural language output.

Loads a fine-tuned LLM (via HuggingFace Transformers) for parsing user prompts and generating function calls.

Integrates with MCP server to execute backend tools (e.g., geospatial analysis).

Uses LangChain and a secondary LLM (e.g. Ollama, phi3) to verbalize JSON/technical responses as natural human language.

Returns both technical and human-friendly results in a unified JSON response.

Easy to extend with new tools and models.


