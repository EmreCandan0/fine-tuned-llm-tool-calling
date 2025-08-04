from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import requests
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

MODEL_DIR = "merged_model"
TOKENIZER_DIR = "lora_model"

MCP_BASE_URL = "http://127.0.0.1:11436"

app = Flask(__name__)

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(model)
print("Ready!")


def parse_all_tools(response):
    tool_names = re.findall(r'TOOL_NEEDED:\s*(\w+)', response)
    params_list = re.findall(r'PARAMS:\s*(\{.*?\})', response, re.DOTALL)
    tools = []
    for i in range(len(tool_names)):
        tool_name = tool_names[i]
        try:
            params = json.loads(params_list[i]) if i < len(params_list) else None
        except Exception:
            params = None
        tools.append((tool_name, params))
    return tools

def call_mcp_tool(tool_name, params):
    """MCP aracını çağırır - event-stream/JSON response uyumlu"""

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": params
        },
        "id": 1
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    try:
        print(f" MCP çağrısı: {tool_name}")
        resp = requests.post(f"{MCP_BASE_URL}/mcp", json=payload, headers=headers)

        result = parse_mcp_response(resp.text)
        if result is None:
            print(f" MCP Yanıt parse edilemedi: {resp.text}")
            return None

        if "result" in result:
            return result["result"]
        elif "error" in result:
            return result["error"]
        else:
            return result

    except Exception as e:
        print(f" MCP Bağlantı hatası: {e}")


def parse_mcp_response(response_text):
    """
    MCP event-stream (SSE) yanıtlarını JSON olarak parse eder.
    """
    if "data:" in response_text:
        lines = [line for line in response_text.splitlines() if line.startswith("data:")]
        if not lines:
            return None
        json_str = lines[-1][len("data: "):]
        try:
            data = json.loads(json_str)
            return data
        except Exception as e:
            print("JSON parse hatası:", e)
            return None
    else:
        try:
            return json.loads(response_text)
        except Exception as e:
            print("JSON parse hatası:", e)
            return None


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=1e-7,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Tool-calling otomatik parse
    tools = parse_all_tools(response)
    tool_results = []


    llm = Ollama(model="phi3")  # veya kendi LLM'in
    verbalizer_prompt = PromptTemplate(
        input_variables=["json_output"],
        template=(
            "Transform json output below to natural human language\n"
            "{json_output}\n"
            "Keep it simple and understandable"
        ),
    )
    verbalizer_chain = LLMChain(llm=llm, prompt=verbalizer_prompt)

    def verbalize_json(json_output):
        return verbalizer_chain.run(json_output=json.dumps(json_output, ensure_ascii=False, indent=2))

    for tool_name, params in tools:
        tool_result = None
        nlp_result = None
        if tool_name and params:
            tool_result = call_mcp_tool(tool_name, params)
            nlp_result = verbalize_json(tool_result)
        tool_results.append({
            "tool_name": tool_name,
            "params": params,
            "tool_result": tool_result,
            "human_result": nlp_result,
        })

    return jsonify({
        "output": response,
        "tools": tool_results,
    })



if __name__ == "__main__":

    app.run(port=8080, host="0.0.0.0")

