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
