# LangChain + LangGraph 教学文档 — 基于 m-chainflow 项目

## 一、两个框架的定位

```
LangChain: 管"节点怎么干"
  ├── ChatModel.invoke()    调用大模型
  ├── Messages               消息体系（System/Human/AI）
  ├── Callback               流式输出、工具追踪
  ├── PromptTemplate         提示词模板
  ├── VectorStore            向量数据库接口
  ├── Memory                 对话记忆
  └── Agent                  智能体（ReAct/Function Call）

LangGraph: 管"图怎么走"
  ├── StateGraph             有状态图（核心）
  ├── add_node               注册节点
  ├── add_edge               固定边
  ├── add_conditional_edges  条件边（动态路由）
  ├── compile                编译图（checkpointer + interrupt）
  ├── stream                 流式执行
  └── interrupt_before       人机交互中断/恢复
```

## 二、项目中的完整工作流

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   意图识别    │  ← LangChain: ChatModel.invoke()
                    └──────┬───────┘
                           │
                  conditional_edges（LangGraph 条件路由）
                  ┌────────┼────────┐
                  │        │        │
                  ▼        ▼        ▼
           ┌──────────┐ ┌─────┐ ┌──────┐
           │ RAG 检索  │ │报名 │ │ 闲聊  │
           │ +LLM生成  │ │流程 │ │ LLM  │
           └─────┬────┘ └──┬──┘ └──┬───┘
                 │    interrupt    │
                 │    (暂停等输入)  │
                 │         │       │
                 │    continue     │
                 │         │       │
                 ▼         ▼       ▼
              ┌────────────────────────┐
              │          END           │
              └────────────────────────┘
```

## 三、LangGraph 核心 API 详解

### 3.1 StateGraph — 创建有状态图

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# 定义状态结构，图中所有节点共享这个状态
class WorkflowState(TypedDict):
    query: str          # 用户问题
    intent: str         # 识别的意图
    answer: str         # 生成的回答
    source_docs: list   # 检索来源
    user_input: dict    # 用户表单输入（报名场景）

# 创建图
graph_builder = StateGraph(WorkflowState)
```

**要点**：StateGraph 的泛型参数是 TypedDict，定义了图中流转的状态字段。每个节点读取 state、处理、更新 state。

### 3.2 add_node — 注册节点

```python
# 节点函数签名：接收 state dict，返回更新后的 state dict
def intent_node(state: dict) -> dict:
    # ... 业务逻辑 ...
    state['intent'] = '课程咨询'
    return state

# 注册到图
graph_builder.add_node("intent", intent_node)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("register", register_node)
graph_builder.add_node("chat", chat_node)
```

**要点**：节点就是一个普通函数，接收 state 返回 state。名字是字符串，后面连边时用。

### 3.3 add_edge — 固定边

```python
from langgraph.constants import START, END

# START → 意图识别（图的入口）
graph_builder.add_edge(START, "intent")

# 各分支 → END（图的出口）
graph_builder.add_edge("rag", END)
graph_builder.add_edge("register", END)
graph_builder.add_edge("chat", END)
```

**要点**：`add_edge(A, B)` = A 执行完一定执行 B。START/END 是特殊常量。

### 3.4 add_conditional_edges — 条件边（动态路由）

```python
# 路由函数：根据 state 中的 intent 返回下一个节点名
def route_intent(state: dict) -> str:
    intent = state.get('intent', '闲聊')
    return {'课程咨询': 'rag', '报名': 'register', '闲聊': 'chat'}[intent]

# 注册条件边
graph_builder.add_conditional_edges(
    "intent",           # 源节点
    route_intent,       # 路由函数
    {                   # 映射：路由函数返回值 → 目标节点名
        'rag': 'rag',
        'register': 'register',
        'chat': 'chat',
    }
)
```

**要点**：
- 路由函数的返回值必须和 mapping 的 key 一致
- 源节点执行完 → 调路由函数 → 在 mapping 中找到目标 → 执行目标节点
- 这就是主项目中意图识别 → 多分支的实现原理

### 3.5 compile — 编译图

```python
from langgraph.checkpoint.memory import MemorySaver

graph = graph_builder.compile(
    checkpointer=MemorySaver(),          # 状态持久化（每步自动保存）
    interrupt_before=["register"],        # 这些节点执行前自动暂停
)
config = {'configurable': {'thread_id': 'demo'}}
```

**三个关键参数**：
- `checkpointer`：每执行一步自动保存状态到检查点，支持中断后恢复
- `interrupt_before`：执行到这些节点前自动暂停（Human-in-the-Loop）
- `recursion_limit`（通过 config 传）：最大执行步数，防死循环

**主项目中**用自定义的 `RedisSaver` 替代 `MemorySaver`，实现跨进程/跨服务的状态持久化。

### 3.6 stream — 执行图

```python
# 首次执行
for event in graph.stream(input_data, config=config):
    pass  # 每执行完一个节点 yield 一次

# 从中断处恢复（传 None）
for event in graph.stream(None, config=config):
    pass
```

**要点**：`stream()` 按拓扑顺序逐个执行节点。传 `None` 表示从上次 interrupt 处继续。

### 3.7 get_state — 获取图快照

```python
snapshot = graph.get_state(config)
next_nodes = snapshot.next  # 下一步要执行的节点列表

if len(next_nodes) == 0:
    # 图执行完毕
    status = 'SUCCESS'
elif 'register' in next_nodes:
    # 被 interrupt 了，等待用户输入
    status = 'INPUT'
```

### 3.8 interrupt_before — 人机交互

```
用户说："我想报名"
     │
     ▼
意图识别 → intent = "报名" → route_node → register
     │
     ▼
graph.stream() 执行到 register 节点前 → 自动暂停
get_state().next = ["register"]
     │
     ▼
前端弹出表单（姓名、手机号）
用户填写提交
     │
     ▼
graph.update_state(config, {'user_input': {'name': '张三', 'phone': '138xxx'}})
graph.stream(None, config)  ← 从 register 节点继续
     │
     ▼
register 节点读取 state['user_input']，生成确认回复
     │
     ▼
→ END
```

## 四、LangChain 核心 API 详解

### 4.1 Messages — 消息体系

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个意图分类器..."),
    HumanMessage(content="你们有什么课程？"),
]

result = llm.invoke(messages)  # result 是 AIMessage
answer = result.content        # 获取文本
```

所有 LLM 交互都通过 `List[BaseMessage]` 进行。这是 LangChain 的统一消息协议。

### 4.2 ChatModel.invoke — 调用模型

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=base_url,    # 兼容通义千问
    model='qwen-max',
    temperature=0.7,
    streaming=True,
)

config = RunnableConfig(callbacks=[my_callback])
result = llm.invoke(messages, config=config)
```

**要点**：`invoke()` 是同步调用，`ainvoke()` 是异步。`RunnableConfig` 用于注入回调。

### 4.3 Callback — 流式输出

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class StreamCollector(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    # 每生成一个 token 调用一次
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        # 可以在这里推送给前端（SSE/WebSocket）

    # LLM 生成结束时调用
    def on_llm_end(self, response: LLMResult, **kwargs):
        full_text = response.generations[0][0].text

# 使用
collector = StreamCollector()
config = RunnableConfig(callbacks=[collector])
result = llm.invoke(messages, config=config)
```

**主项目中的用法**：`LLMNodeCallbackHandler` 继承 `BaseCallbackHandler`，在 `on_llm_new_token` 中通过 Redis 队列推送 token 到前端 SSE。

### 4.4 VectorStore — 向量检索

```python
# 项目中用简单的内存知识库（SimpleKnowledgeBase）
# 实际等价于：
docs = vectorstore.similarity_search(query, k=2)
# 返回 List[Document]，每个 Document 有 page_content 和 metadata
```

**主项目中**用 `langchain_community.vectorstores.Milvus` 连接真实的 Milvus 向量数据库。

### 4.5 PromptTemplate — 提示词模板

```python
from langchain_core.prompts import ChatPromptTemplate

# RAG 场景：把检索结果嵌入提示词
system_msg = f"""根据以下知识库内容回答用户问题：
{context}"""

messages = [
    SystemMessage(content=system_msg),
    HumanMessage(content=query),
]
```

## 五、LangChain 和 LangGraph 的协作关系

```
┌──────────────────────────────────────────────────────────┐
│              LangGraph（外层编排）                         │
│                                                          │
│  StateGraph → add_node → add_edge → compile → stream     │
│                                                          │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐  ┌─────────┐ │
│  │ Start   │──→│  意图    │──→│  RAG     │──→│  End   │ │
│  └─────────┘   └────┬────┘   └────┬─────┘  └─────────┘ │
│                      │             │                     │
└──────────────────────┼─────────────┼─────────────────────┘
                       │             │
┌──────────────────────┼─────────────┼─────────────────────┐
│              LangChain（内层执行）   │                     │
│                      │             │                     │
│                      ▼             ▼                     │
│   ┌──────────────────────┐ ┌──────────────────────────┐  │
│   │ SystemMessage        │ │ VectorStore.search()     │  │
│   │ HumanMessage         │ │ PromptTemplate           │  │
│   │ llm.invoke(messages) │ │ llm.invoke(messages)     │  │
│   │ RunnableConfig       │ │ Callback (流式输出)       │  │
│   └──────────────────────┘ └──────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**一句话**：LangGraph 调度节点执行顺序（图怎么走），LangChain 在每个节点内部调 LLM / 检索 / Agent（节点怎么干）。二者通过节点函数 `run(state) -> state` 连接。

## 六、项目中的关键实现细节

### 6.1 通义千问适配

LlamaIndex 用 `OpenAILike`（不校验模型名），LangChain 的 `ChatOpenAI` 天然支持 `openai_api_base` 参数，直接兼容通义千问：

```python
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    model='qwen-max',
)
```

### 6.2 StepTracer 可视化机制

每个关键操作调用 `tracer.trace()`，记录代码片段 + 输入输出 + 教学说明，通过 asyncio.Queue → SSE 推送到前端：

```python
tracer.trace(
    phase='execute',
    title='意图识别 — 调用 LLM',
    code='▶ result = llm.invoke(messages, config=config)',
    input_data={'messages_count': 2},
    output_data={'intent': '课程咨询'},
    explanation='invoke() 是 LangChain 调用模型的核心方法...',
    component='LangChain:ChatModel.invoke',
)
```

### 6.3 主项目中的对应关系

| Demo 中的简化实现 | 主项目（ZFAPT）中的完整实现 |
|------------------|--------------------------|
| `MemorySaver` 内存检查点 | `RedisSaver` Redis 持久化检查点 |
| `SimpleKnowledgeBase` 内存向量库 | Milvus + ES 双路存储 |
| `route_intent()` 简单路由 | 意图识别节点（LLM 判断）+ 条件边 |
| 4 个硬编码节点 | 17 种节点类型 + `NodeFactory` 工厂模式 |
| `StreamCollector` 收集 token | `LLMNodeCallbackHandler` → Redis → SSE |
| `graph.stream()` 同步执行 | `ThreadPoolExecutor` 异步 + `RedisCallback` 事件推送 |
