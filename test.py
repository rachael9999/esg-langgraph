import requests
headers = {"Authorization":"Bearer sk-ccd47e9a930a4c29a36b0dd16be45dde","Content-Type":"application/json"}
body = {
  "model": "qwen-flash",
  "messages": [
    {"role":"user", "content":[{"type":"text","text":"Hello"}]}
  ]
}
r = requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                  headers=headers, json=body, timeout=60)
print(r.status_code, r.text)