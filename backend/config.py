# AI_GENERATE_START
import os
from pathlib import Path
from dotenv import load_dotenv

# 从项目根目录加载 .env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / '.env')


class LLMConfig:
    """LLM 配置，从 .env 读取"""
    api_key: str = os.getenv('LLM_API_KEY', '')
    base_url: str = os.getenv('LLM_BASE_URL', '')
    model: str = os.getenv('LLM_MODEL', 'qwen-max')
    temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))


llm_config = LLMConfig()
# AI_GENERATE_END
