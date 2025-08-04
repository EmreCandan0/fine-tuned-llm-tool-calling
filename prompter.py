import requests


url = 'http://127.0.0.1:8080/generate'
def main():

    while True:
        user_input = input("➤ Talebiniz: ").strip()
        prompt = {'prompt': f'<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n'}
        response = requests.post(url, json=prompt)
        data = response.json()

        for tool in data.get("tools", []):
            human_result = tool.get("human_result")
            if human_result:
                print("🟢", human_result)
            else:
                print("⚠️ Teknik sonuç:", tool.get("tool_result"))


if __name__ == "__main__":
    main()
