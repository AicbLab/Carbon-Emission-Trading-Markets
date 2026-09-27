"""
LLM客户端模块
支持自定义API Key，提供统一的LLM调用接口
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 尝试导入openai
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: 未安装openai库，请运行: pip install openai")

from .config import config


@dataclass
class LLMResponse:
    """LLM响应封装"""
    content: str
    parsed_json: Optional[Dict] = None
    success: bool = True
    error_message: str = ""
    tokens_used: int = 0


class LLMClient:
    """
    LLM客户端类
    支持多种API配置，提供同步和异步调用接口
    """
    
    def __init__(self, api_key: str = "", base_url: str = "", model: str = ""):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥，如未提供则从配置或环境变量读取
            base_url: API基础URL
            model: 模型名称
        """
        self.api_key = api_key or config.llm.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or config.llm.base_url
        self.model = model or config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        if OPENAI_AVAILABLE and self.api_key:
            self._init_clients()
    
    def _init_clients(self):
        """初始化OpenAI客户端"""
        if not OPENAI_AVAILABLE:
            return
            
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def is_configured(self) -> bool:
        """检查是否已配置API Key"""
        return bool(self.api_key) and OPENAI_AVAILABLE
    
    def setup_api_key(self, api_key: str):
        """
        设置API Key（交互式输入接口）
        
        Args:
            api_key: 用户提供的API Key
        """
        self.api_key = api_key
        config.llm.api_key = api_key
        if OPENAI_AVAILABLE:
            self._init_clients()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """
        同步聊天补全
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式，如 "json_object"
        
        Returns:
            LLMResponse对象
        """
        if not self.is_configured():
            return LLMResponse(
                content="",
                success=False,
                error_message="LLM未配置，请先设置API Key"
            )
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            if response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            
            completion = self._client.chat.completions.create(**kwargs)
            
            content = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else 0
            
            # 尝试解析JSON
            parsed_json = None
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                pass
            
            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                success=True,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error_message=str(e)
            )
    
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """
        异步聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式
        
        Returns:
            LLMResponse对象
        """
        if not self.is_configured():
            return LLMResponse(
                content="",
                success=False,
                error_message="LLM未配置，请先设置API Key"
            )
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            if response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            
            completion = await self._async_client.chat.completions.create(**kwargs)
            
            content = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else 0
            
            # 尝试解析JSON
            parsed_json = None
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                pass
            
            return LLMResponse(
                content=content,
                parsed_json=parsed_json,
                success=True,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error_message=str(e)
            )
    
    async def batch_chat_completion(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> List[LLMResponse]:
        """
        批量异步聊天补全
        
        Args:
            batch_messages: 多组消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式
        
        Returns:
            LLMResponse对象列表
        """
        tasks = [
            self.chat_completion_async(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            for messages in batch_messages
        ]
        return await asyncio.gather(*tasks)
    
    def generate_decision(
        self,
        system_prompt: str,
        user_prompt: str,
        require_json: bool = True,
    ) -> LLMResponse:
        """
        生成决策（专门用于Agent决策）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            require_json: 是否要求JSON格式输出
        
        Returns:
            LLMResponse对象
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return self.chat_completion(
            messages=messages,
            response_format="json_object" if require_json else None,
        )


# 全局LLM客户端实例
llm_client = LLMClient()


def setup_llm_interactive():
    """
    交互式设置LLM API Key
    用户可以在此输入自己的API Key
    """
    print("=" * 60)
    print("碳金融市场ABM仿真 - LLM API配置")
    print("=" * 60)
    print()
    print("请配置您的大语言模型API信息:")
    print("(支持阿里云百炼、OpenAI等兼容OpenAI API格式的服务)")
    print()
    
    # 检查是否已有环境变量配置
    env_key = os.getenv("DASHSCOPE_API_KEY", "")
    if env_key:
        print(f"检测到环境变量 DASHSCOPE_API_KEY 已设置")
        use_env = input("是否使用环境变量中的API Key? (y/n): ").strip().lower()
        if use_env == 'y':
            llm_client.setup_api_key(env_key)
            print("✓ 已使用环境变量中的API Key")
            return True
    
    # 手动输入API Key
    print()
    print("请输入您的API Key (输入空字符串则跳过LLM功能):")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("⚠ 未提供API Key，LLM功能将被禁用，使用规则决策代替")
        return False
    
    # 输入Base URL (可选)
    print()
    default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    print(f"请输入API Base URL (直接回车使用默认值: {default_url}):")
    base_url = input("Base URL: ").strip()
    if not base_url:
        base_url = default_url
    
    # 输入模型名称 (可选)
    print()
    default_model = "qwen-plus"
    print(f"请输入模型名称 (直接回车使用默认值: {default_model}):")
    model = input("Model: ").strip()
    if not model:
        model = default_model
    
    # 配置客户端
    llm_client.setup_api_key(api_key)
    llm_client.base_url = base_url
    llm_client.model = model
    llm_client._init_clients()
    
    print()
    print("=" * 60)
    print("✓ LLM API配置完成!")
    print(f"  - Model: {model}")
    print(f"  - Base URL: {base_url}")
    print("=" * 60)
    
    return True


# 便捷函数
def get_llm_client() -> LLMClient:
    """获取全局LLM客户端实例"""
    return llm_client
