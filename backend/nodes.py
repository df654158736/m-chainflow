# AI_GENERATE_START
"""
工作流节点实现
每个节点都是一个函数，接收 state dict，返回更新后的 state dict
在关键操作处调用 tracer.trace() 记录步骤
"""
import json
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from backend.config import llm_config
from backend.knowledge_base import knowledge_base
from backend.step_tracer import StepTracer


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """获取 LLM 实例"""
    return ChatOpenAI(
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.base_url,
        model=llm_config.model,
        temperature=llm_config.temperature,
        streaming=streaming,
    )


# ============================================================
# 意图识别节点
# ============================================================
INTENT_SYSTEM_PROMPT = """你是一个意图分类器。根据用户输入，判断用户意图属于以下三类之一：
1. "课程咨询" — 用户询问课程信息、培训内容、时间、费用等
2. "报名" — 用户想要报名、注册、参加某个课程
3. "闲聊" — 其他不相关的对话

请严格以 JSON 格式返回，不要返回其他内容：
{"intent": "课程咨询"} 或 {"intent": "报名"} 或 {"intent": "闲聊"}"""


def intent_node(state: dict, tracer: StepTracer) -> dict:
    """意图识别节点"""
    query = state['query']
    llm = get_llm(streaming=False)

    # 步骤：构建消息列表
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    tracer.trace(
        phase='execute', title='意图识别 — 构建消息列表',
        code=(
            'messages = [\n'
            '    SystemMessage(content="你是一个意图分类器..."),\n'
            '    HumanMessage(content=f"{query}")\n'
            ']'
        ),
        node_id='intent', node_status='running',
        input_data={'system_prompt': INTENT_SYSTEM_PROMPT[:80] + '...', 'user_query': query},
        explanation=(
            'LangChain 的消息体系：所有 LLM 交互都通过 List[BaseMessage] 进行。\n'
            'SystemMessage 设定模型角色，HumanMessage 传入用户输入。'
        ),
        component='LangChain:Messages',
    )

    # 步骤：调用 LLM
    config = RunnableConfig()
    result = llm.invoke(messages, config=config)

    # 解析意图
    try:
        content = result.content.replace('```json', '').replace('```', '').strip()
        intent_data = json.loads(content)
        intent = intent_data.get('intent', '闲聊')
    except (json.JSONDecodeError, AttributeError):
        intent = '闲聊'

    tracer.trace(
        phase='execute', title='意图识别 — 调用 LLM 并解析结果',
        code=(
            'config = RunnableConfig()\n'
            '▶ result = llm.invoke(messages, config=config)\n'
            'intent = json.loads(result.content)["intent"]'
        ),
        node_id='intent', node_status='completed',
        input_data={'messages_count': len(messages)},
        output_data={'intent': intent, 'raw_response': result.content[:200]},
        explanation=(
            'invoke() 是 LangChain 调用模型的核心方法。\n'
            '传入 List[BaseMessage]，返回 AIMessage。\n'
            'RunnableConfig 可以传入 callbacks 实现流式输出等功能。'
        ),
        component='LangChain:ChatModel.invoke',
    )

    state['intent'] = intent
    return state


# ============================================================
# 意图路由函数
# ============================================================
def route_intent(state: dict) -> str:
    """条件边路由函数 — 根据意图返回下一个节点 ID"""
    intent = state.get('intent', '闲聊')
    mapping = {
        '课程咨询': 'rag',
        '报名': 'register',
        '闲聊': 'chat',
    }
    return mapping.get(intent, 'chat')


# ============================================================
# RAG 检索 + LLM 生成节点
# ============================================================
class StreamCollector(BaseCallbackHandler):
    """收集流式 token 的回调处理器"""

    def __init__(self):
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass


RAG_SYSTEM_PROMPT = """你是一个专业的课程咨询顾问。请根据以下检索到的知识库内容回答用户的问题。
如果知识库内容不足以回答，请如实说明。

知识库内容：
{context}"""


def rag_node(state: dict, tracer: StepTracer) -> dict:
    """RAG 检索 + LLM 生成节点"""
    query = state['query']

    # 步骤：向量检索
    docs = knowledge_base.search(query, top_k=2)
    tracer.trace(
        phase='execute', title='RAG — 向量检索知识库',
        code=(
            '# 使用 VectorStore 检索相关文档\n'
            '▶ docs = vectorstore.similarity_search(\n'
            '    query=query, k=2\n'
            ')'
        ),
        node_id='rag', node_status='running',
        input_data={'query': query, 'top_k': 2},
        output_data={'retrieved_docs': [{'content': d['content'][:80] + '...', 'score': d['score']} for d in docs]},
        explanation=(
            'VectorStore.similarity_search() 是 LangChain 向量检索的核心方法。\n'
            '将用户 query 通过 Embedding 模型编码为向量，\n'
            '在向量数据库中找到最相似的 top_k 条文档。'
        ),
        component='LangChain:VectorStore',
    )

    # 步骤：构建带上下文的 Prompt
    context = '\n\n'.join([d['content'] for d in docs])
    system_msg = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=query),
    ]
    tracer.trace(
        phase='execute', title='RAG — 构建检索增强 Prompt',
        code=(
            '# 将检索结果嵌入 Prompt 模板\n'
            'context = "\\n".join([d.page_content for d in docs])\n'
            '▶ messages = [\n'
            '    SystemMessage(content=f"根据知识库内容：{context}..."),\n'
            '    HumanMessage(content=query)\n'
            ']'
        ),
        node_id='rag', node_status='running',
        input_data={'context_length': len(context), 'query': query},
        explanation=(
            'RAG 的核心思路：将检索到的文档内容拼接到 SystemMessage 中，\n'
            '让 LLM 基于这些上下文来回答问题，减少幻觉。\n'
            'ChatPromptTemplate 可以用模板变量自动填充。'
        ),
        component='LangChain:PromptTemplate',
    )

    # 步骤：流式调用 LLM
    llm = get_llm(streaming=True)
    collector = StreamCollector()
    config = RunnableConfig(callbacks=[collector])
    result = llm.invoke(messages, config=config)
    answer = result.content

    tracer.trace(
        phase='execute', title='RAG — 流式调用 LLM 生成回答',
        code=(
            '# 配置流式回调\n'
            'collector = StreamCollector()  # 继承 BaseCallbackHandler\n'
            'config = RunnableConfig(callbacks=[collector])\n'
            '\n'
            '▶ result = llm.invoke(messages, config=config)\n'
            '\n'
            '# collector.on_llm_new_token() 每生成一个 token 被调用一次'
        ),
        node_id='rag', node_status='completed',
        input_data={'messages_count': len(messages)},
        output_data={'answer': answer[:200] + ('...' if len(answer) > 200 else ''), 'token_count': len(collector.tokens)},
        explanation=(
            'Callback 是 LangChain 的核心机制之一。\n'
            '继承 BaseCallbackHandler，重写 on_llm_new_token() 可以拦截每个生成的 token。\n'
            '在主项目中，这个机制用于将流式 token 通过 Redis 推送给前端 SSE。\n'
            'RunnableConfig(callbacks=[...]) 将回调注入到 invoke 调用中。'
        ),
        component='LangChain:Callback',
    )

    state['answer'] = answer
    state['source_docs'] = [d['content'] for d in docs]
    return state


# ============================================================
# 报名流程节点
# ============================================================
def register_node(state: dict, tracer: StepTracer) -> dict:
    """报名流程节点 — 生成确认回复"""
    user_input = state.get('user_input', {})
    name = user_input.get('name', '未提供')
    phone = user_input.get('phone', '未提供')

    llm = get_llm(streaming=False)
    messages = [
        SystemMessage(content='你是课程报名助手，请根据用户提供的信息生成报名确认回复。'),
        HumanMessage(content=f'用户 {name}（手机号：{phone}）想要报名课程。请生成一条简短的报名确认消息。'),
    ]

    tracer.trace(
        phase='execute', title='报名 — 接收用户输入并生成确认',
        code=(
            '# interrupt 恢复后，从 state 获取用户输入\n'
            'user_input = state["user_input"]\n'
            'name = user_input["name"]\n'
            'phone = user_input["phone"]\n'
            '\n'
            '▶ result = llm.invoke([\n'
            '    SystemMessage(content="你是课程报名助手..."),\n'
            '    HumanMessage(content=f"用户{name}想要报名...")\n'
            '])'
        ),
        node_id='register', node_status='running',
        input_data={'name': name, 'phone': phone},
        explanation=(
            '这是 Human-in-the-Loop 恢复后的节点。\n'
            'LangGraph 的 interrupt_before 暂停图执行，\n'
            '用户提交表单后调用 continue_run(None) 恢复，\n'
            '此时 state 中已包含用户填写的数据。'
        ),
        component='LangChain:ChatModel.invoke',
    )

    result = llm.invoke(messages)
    answer = result.content

    tracer.trace(
        phase='execute', title='报名 — 生成确认回复完成',
        code='answer = result.content',
        node_id='register', node_status='completed',
        output_data={'answer': answer[:200]},
        component='LangChain:ChatModel.invoke',
    )

    state['answer'] = answer
    return state


# ============================================================
# 闲聊节点
# ============================================================
def chat_node(state: dict, tracer: StepTracer) -> dict:
    """闲聊节点"""
    query = state['query']
    llm = get_llm(streaming=True)

    messages = [
        SystemMessage(content='你是一个友好的客服助手，请自然地回复用户。'),
        HumanMessage(content=query),
    ]

    tracer.trace(
        phase='execute', title='闲聊 — 直接调用 LLM',
        code=(
            'messages = [\n'
            '    SystemMessage(content="你是一个友好的客服助手..."),\n'
            '    HumanMessage(content=query)\n'
            ']\n'
            '▶ result = llm.invoke(messages)'
        ),
        node_id='chat', node_status='running',
        input_data={'query': query},
        explanation=(
            '最简单的 LangChain 调用方式：\n'
            '构建消息列表 → invoke → 获取回复。\n'
            '没有 RAG 检索，没有工具调用，纯对话。'
        ),
        component='LangChain:ChatModel.invoke',
    )

    collector = StreamCollector()
    config = RunnableConfig(callbacks=[collector])
    result = llm.invoke(messages, config=config)
    answer = result.content

    tracer.trace(
        phase='execute', title='闲聊 — 回复生成完成',
        code='answer = result.content',
        node_id='chat', node_status='completed',
        output_data={'answer': answer[:200] + ('...' if len(answer) > 200 else '')},
        component='LangChain:ChatModel.invoke',
    )

    state['answer'] = answer
    return state
# AI_GENERATE_END
