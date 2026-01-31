import http.client
from src.llm.config import API_KEY

# 创建 HTTPS 连接（注意：http.client 不接受 URL 协议前缀）
conn = http.client.HTTPConnection("172.21.15.28", 80)

payload = "{\n    \"model\": \"gemini-2.5-flash\",\n    \"messages\": [\n      {\"role\": \"user\", \"content\": \"Hello, how are you?\"}\n    ]\n  }"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# 发送 POST 请求
conn.request("POST", "/v1/chat/completions", payload, headers)

# 获取响应
res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
