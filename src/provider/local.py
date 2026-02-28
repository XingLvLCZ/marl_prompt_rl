import re
import requests
from typing import List, Dict, Optional, Union


class LocalProvider:
    def __init__(self, api_key: str = "", base_url: Optional[str] = None, model: str = "default"):
        """
        初始化 LocalProvider，用于调用本地 vLLM 服务（OpenAI 兼容接口）
        
        Args:
            api_key: API密钥（本地服务通常不需要，但保留参数以保持兼容）
            base_url: API基础URL，例如 "http://localhost:8000/v1"
                     如果未提供，默认使用 "http://localhost:8000/v1"
            model: 模型名称，如果本地服务只有一个模型，可设为任意值，服务会自动使用已加载的模型
        """
        self.api_key = api_key
        # 拼接完整的聊天补全端点
        if base_url:
            # 确保 base_url 以 /v1 结尾，如果没有则补上
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:8000/v1"
        self.model = model if model != "default" else self._get_default_model()

    def call(self, 
             messages: Union[str, List[Dict[str, str]]],
             strip_think: bool = True,   # 默认开启去除 think 内容
             **kwargs) -> str:
        """
        调用本地 vLLM 服务
        
        Args:
            messages: 消息内容，可以是字符串或符合 OpenAI 格式的消息列表
            strip_think: 是否去除 ... 标签及其内容
            **kwargs: 其他 API 参数，如 temperature, max_tokens, top_p 等
            
        Returns:
            模型生成的文本内容（已去除首尾空白，并根据 strip_think 清理 think 内容）
        """
        # 如果是字符串，转换为单条用户消息
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 准备请求头（本地服务可能不需要 Authorization，但保留以兼容某些场景）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 准备请求体
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,          # 确保非流式响应
            **kwargs
        }

        try:
            response = requests.post(self.base_url + "/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # 提取回复内容（兼容 OpenAI 格式）
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                # print(len(content))
                
                # 后处理：去除 think 标签及其内部内容
                if strip_think:
                    # 匹配常见的几种思考格式（可根据实际输出调整）
                    
                    # 匹配  ... 
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    
                    # 匹配 思考：... （如果思考内容以特定文字开头）
                    # 注意：这种格式较难完美去除，因为可能误伤正常内容
                    # content = re.sub(r'思考：.*?\n', '', content)
                    
                    # 清理可能留下的多余空行和空格
                    content = re.sub(r'\n\s*\n', '\n', content).strip()
                
                return content.strip()
            else:
                raise ValueError("API 响应格式异常：缺少 'choices' 字段或为空")

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"本地服务请求失败: {e}")
        except KeyError as e:
            raise KeyError(f"解析 API 响应时出错: {e}")

    def __call__(self, *args, **kwargs):
        """
        使实例可调用，直接转发到 call 方法
        """
        return self.call(*args, **kwargs)
    
    def _get_default_model(self) -> str:
        """从 /v1/models 接口获取第一个模型名称"""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            # print(f"Available models from local vLLM server: {[m['id'] for m in models]}")
            if not models:
                raise ValueError("No models available from vLLM server")
            return models[0]["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch model list: {e}")


# 使用示例
if __name__ == "__main__":
    # 初始化 provider（假设本地 vLLM 服务运行在默认地址）
    provider = LocalProvider(
        api_key="not-needed",          # 本地服务通常不需要，但保留参数
        base_url="http://localhost:8000/v1",  # 可省略，默认即为该值
        model="Qwen/Qwen3-14B"
    )

    # 方式1：直接传入字符串
    response1 = provider("你好，请介绍一下自己")
    print(f"Response 1:\n{response1}\n")

    # 方式2：使用消息列表
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "今天天气怎么样？"}
    ]
    response2 = provider.call(messages, temperature=0.8, max_tokens=200)
    print(f"Response 2:\n{response2}\n")

    # 方式3：通过实例直接调用并传入额外参数
    response3 = provider("""
Generate a collaboration protocol for solving GSM8K-style grade-school arithmetic word problems that prioritizes adaptive, informative, and concise communication among the strategist, calculator, and verifier agents.  

The protocol should ensure that each agent's output is contextually relevant, responsive to the current state of subgoals, and aligned with the overall objective of maximizing final answer correctness.  

- The strategist should decompose the problem into manageable subgoals, propose a clear and feasible plan, and clearly identify unresolved subgoals to guide subsequent steps. Communicate with clarity and precision, adjusting tone and level of detail based on the calculator's needs and the verifier's feedback.  
- The calculator should interpret the strategist's instructions, perform arithmetic steps explicitly, and return only necessary, well-structured equations and intermediate results. It should adapt its level of detail—being thorough when needed, concise otherwise—and signal when it requires clarification or additional context.  
- The verifier should assess consistency, logical coherence, and numerical accuracy, and communicate appropriately: confirm correctness, flag inconsistencies, or request clarification or re-computation. It should be responsive to both the strategist’s plan and the calculator’s output, adjusting its feedback to support progress without overcomplicating or over-explaining.  

Communication must be adaptive: shift in tone, detail, and focus based on the agent’s role, the stage of problem-solving, and the progress of subgoals. Avoid rigid formatting or templates—instead, allow natural, goal-directed language that supports clarity, efficiency, and error correction. Prioritize meaningful, actionable information over formality or strict structure.  

Ensure all interactions are concise, directly relevant, and designed to resolve subgoals or enable final validation—never introducing unnecessary steps or redundant explanations.
""", temperature=0.9)
    print(f"Response 3:\n{response3}")