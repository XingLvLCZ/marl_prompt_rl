import requests
from typing import List, Dict, Optional, Union


class DeepSeekProvider:
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = 'deepseek-ai/DeepSeek-V3'):
        """
        初始化 DeepSeekProvider
        
        Args:
            api_key: API密钥
            base_url: API基础URL（可选，默认为本地地址）
        """
        self.api_key = api_key
        self.base_url = base_url or 'https://localhost:8000/v1/chat/completions'
        self.model = model
        
    def call(self, 
             messages: Union[str, List[Dict[str, str]]],
             **kwargs) -> str:
        """
        调用DeepSeek API
        
        Args:
            messages: 消息内容，可以是字符串或消息列表
            model: 模型名称
            **kwargs: 其他API参数
            
        Returns:
            API返回的文本内容
            
        Raises:
            requests.exceptions.RequestException: 网络请求异常
            KeyError: API响应格式异常
        """
        # 如果是字符串，转换为消息列表
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        
        # 准备请求头
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 准备请求体
        payload = {
            'model': self.model,
            'messages': messages,
            **kwargs  # 允许传入其他参数
        }
        
        try:
            # 发送请求
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # 检查HTTP错误
            
            # 解析响应
            result = response.json()
            
            # 提取回复内容
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                raise ValueError("API响应格式异常，未找到choices字段")
                
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"API请求失败: {e}")
        except KeyError as e:
            raise KeyError(f"解析API响应时出错: {e}")
    
    def __call__(self, *args, **kwargs):
        """
        使实例可调用，直接转发到call方法
        """
        return self.call(*args, **kwargs)


# 使用示例
if __name__ == "__main__":
    from config import API_KEY
    # 初始化provider
    provider = DeepSeekProvider(
        api_key=API_KEY,
        base_url="http://172.21.15.28:80/v1/chat/completions",
        model="deepseek-ai/DeepSeek-V3.1"
    )
    
    # 方式1: 直接调用实例
    response1 = provider("how are you?")
    print(f"Response 1: {response1}")
    
    # # 方式2: 使用call方法
    # response2 = provider.call("what is deepseek?")
    # print(f"Response 2: {response2}")
    
    # # 方式3: 使用完整的消息列表
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "tell me a joke"}
    # ]
    # response3 = provider.call(messages)
    # print(f"Response 3: {response3}")
    
    # # 方式4: 传入额外参数
    # response4 = provider.call(
    #     "explain quantum physics",
    #     temperature=0.7,
    #     max_tokens=500
    # )
    # print(f"Response 4: {response4}")