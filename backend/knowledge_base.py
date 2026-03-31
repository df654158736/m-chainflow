# AI_GENERATE_START
import math
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from backend.config import llm_config


# 预置的课程知识文档
KNOWLEDGE_DOCS = [
    '我们提供三门课程：Python入门、AI大模型应用开发、数据分析实战。'
    'Python入门适合零基础学员，AI大模型课程面向有编程经验的开发者，数据分析课程适合数据相关岗位人员。',

    'Python入门课程：每周一到周五 9:00-12:00，共4周，培训费用3000元。'
    '上课地点：A栋3楼301教室。报名链接：https://example.com/python',

    'AI大模型应用开发课程：每周六 14:00-18:00，共8周，培训费用5000元。'
    '上课地点：B栋5楼实验室。报名链接：https://example.com/ai-llm。'
    '课程内容包括：LangChain框架、RAG检索增强生成、LangGraph工作流编排、Agent智能体开发。',
]


class SimpleKnowledgeBase:
    """简单的内存向量知识库"""

    def __init__(self):
        self._docs = KNOWLEDGE_DOCS
        self._embeddings: Optional[list[list[float]]] = None
        self._embedding_model = None

    def initialize(self):
        """初始化知识库，对文档进行 Embedding 编码"""
        try:
            self._embedding_model = OpenAIEmbeddings(
                openai_api_key=llm_config.api_key,
                openai_api_base=llm_config.base_url,
                model='text-embedding-v3',
            )
            self._embeddings = self._embedding_model.embed_documents(self._docs)
        except Exception:
            # Embedding 不可用时使用简单关键词匹配
            self._embedding_model = None
            self._embeddings = None

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        """检索与 query 最相关的文档"""
        if self._embedding_model and self._embeddings:
            return self._vector_search(query, top_k)
        return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """基于向量相似度检索"""
        query_vec = self._embedding_model.embed_query(query)
        scores = []
        for i, doc_vec in enumerate(self._embeddings):
            score = _cosine_similarity(query_vec, doc_vec)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            results.append({
                'content': self._docs[i],
                'score': round(score, 4),
            })
        return results

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """回退方案：简单关键词匹配"""
        scores = []
        for i, doc in enumerate(self._docs):
            # 计算查询词与文档的字符重叠度
            overlap = sum(1 for c in query if c in doc)
            scores.append((i, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            results.append({
                'content': self._docs[i],
                'score': round(score / max(len(query), 1), 4),
            })
        return results


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# 全局知识库实例
knowledge_base = SimpleKnowledgeBase()
# AI_GENERATE_END
