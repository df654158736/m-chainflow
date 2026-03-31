# LangChain + LangGraph 面试问答 — 基于 m-chainflow 项目实战

## 一、LangGraph 基础

### Q1：LangGraph 是什么？和 LangChain 是什么关系？

LangGraph 是 LangChain 团队推出的**图编排框架**，专门用于构建有状态的、多步骤的 AI 工作流。

```
LangChain: 提供 LLM 调用、RAG 检索、Agent 等基础组件
LangGraph: 在 LangChain 之上，提供图编排能力（节点调度、条件路由、状态管理、中断恢复）
```

类比：LangChain 是砖头（组件），LangGraph 是建筑师（把砖头搭成房子）。

### Q2：LangGraph 的 StateGraph 是什么？

StateGraph 是 LangGraph 的核心类，一个**有向图**，所有节点共享一个 TypedDict 状态：

```python
class WorkflowState(TypedDict):
    query: str
    intent: str
    answer: str

graph = StateGraph(WorkflowState)
```

每个节点函数接收 state、处理、更新 state 返回。图按拓扑顺序逐个执行节点，state 在节点之间流转。

### Q3：add_edge 和 add_conditional_edges 有什么区别？

```
add_edge(A, B):
  A 执行完 → 一定执行 B（固定路径）

add_conditional_edges(A, route_fn, mapping):
  A 执行完 → 调 route_fn(state) → 返回值在 mapping 中找到目标 → 执行目标
  （动态路径，运行时根据 state 决定走哪条）
```

**项目中的使用**：意图识别后用 `add_conditional_edges` 根据意图分三路（RAG/报名/闲聊），这就是主项目中 17 种节点类型动态路由的简化版。

### Q4：interrupt_before 是怎么实现人机交互的？

```
compile 时注册: interrupt_before=["register"]

执行到 register 前 → 图自动暂停 → get_state().next = ["register"]
前端弹出表单 → 用户填写提交
graph.update_state(config, {'user_input': {...}})  ← 更新状态
graph.stream(None, config)  ← 传 None 表示从检查点恢复

恢复后 register 节点从 state 中读取 user_input，继续执行
```

**主项目中的使用**：工作流的 Input 节点和 Output 节点都用 `interrupt_before` 实现等待用户输入。配合 `RedisSaver` 持久化检查点，支持跨进程/跨服务恢复。

### Q5：checkpointer 有什么用？为什么主项目要自己实现 RedisSaver？

checkpointer 在每个节点执行完后自动保存图的状态快照（当前在哪个节点、state 的值等），用于：

1. **中断恢复**：interrupt 暂停后，从检查点恢复继续执行
2. **错误重试**：节点执行失败后，从上一个检查点重试
3. **状态查询**：`get_state()` 查看当前图的执行状态

LangGraph 内置的 `MemorySaver` 是内存版本，重启就没了。**主项目自实现了 `RedisSaver`**，原因是：

- 生产环境需要持久化（服务重启后工作流能恢复）
- 多 Worker 共享状态（Celery Worker 之间通过 Redis 同步）
- TTL 过期清理（超时的工作流自动清除）

### Q6：主项目的 GraphEngine 和 Demo 的 WorkflowEngine 有什么区别？

```
Demo (WorkflowEngine):                 主项目 (GraphEngine):
  4 个硬编码节点                          17 种节点类型 + NodeFactory 工厂
  一个固定的工作流                        从 JSON 动态构建任意工作流
  MemorySaver 内存检查点                  RedisSaver Redis 检查点
  同步执行                               ThreadPoolExecutor 异步
  无事件回调                             RedisCallback → SSE 推送

核心原理一样：StateGraph → add_node → add_edge → compile → stream
区别在于 Demo 是静态写死的，主项目是从前端 ReactFlow JSON 动态编译的
```

---

## 二、LangChain 基础

### Q7：LangChain 的 Messages 体系是什么？

LangChain 统一用 `List[BaseMessage]` 和 LLM 交互：

```python
messages = [
    SystemMessage(content="你是客服助手"),   # 系统角色设定
    HumanMessage(content="你好"),           # 用户消息
    AIMessage(content="你好！有什么需要？"),  # AI 回复（多轮时）
    HumanMessage(content="你们有什么课程？"), # 用户追问
]
result = llm.invoke(messages)  # result 是 AIMessage
```

不同 LLM 提供商（OpenAI/Claude/通义千问）内部格式不同，LangChain 用 Messages 做了统一抽象。

### Q8：invoke 和 stream 有什么区别？

```python
# invoke: 同步，等 LLM 完整生成后一次性返回
result = llm.invoke(messages)  # 返回 AIMessage

# stream: 流式，边生成边返回 token
for chunk in llm.stream(messages):
    print(chunk.content, end='')  # 逐 token 输出
```

**项目中**用 `invoke` + `Callback` 的方式实现流式，因为 `on_llm_new_token` 回调在 `invoke` 内部也会触发（只要 `streaming=True`）。

### Q9：Callback 机制是怎么工作的？

```python
class MyCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # 每生成一个 token 调用（流式核心）
    def on_llm_end(self, response, **kwargs):
        # 生成完毕调用
    def on_tool_start(self, serialized, input_str, **kwargs):
        # Agent 工具开始调用
    def on_tool_end(self, output, **kwargs):
        # Agent 工具调用结束

# 注入到调用链
config = RunnableConfig(callbacks=[MyCallback()])
result = llm.invoke(messages, config=config)
```

**主项目中的用法**：
- `LLMNodeCallbackHandler.on_llm_new_token` → Redis 队列 → SSE 推送到前端（Token 级流式输出）
- `on_tool_start/end` → 追踪 Agent 的工具调用过程
- `on_llm_end` → 提取 `reasoning_content`（深度推理内容）

### Q10：RunnableConfig 是什么？

RunnableConfig 是 LangChain 的配置注入机制，用于把 callbacks、metadata、tags 等传递给调用链中的每个组件：

```python
config = RunnableConfig(
    callbacks=[stream_callback, log_callback],  # 回调列表
    metadata={'user_id': '123'},                # 元数据
    tags=['production'],                        # 标签
)
# 传给 invoke，内部所有组件都能拿到
result = llm.invoke(messages, config=config)
```

---

## 三、RAG 检索

### Q11：项目中的 RAG 是怎么做的？

```
用户问："你们有什么课程？"
     │
     ▼
① VectorStore 检索
   query → Embedding → 向量 → 余弦相似度 top-k
   返回最相关的 2 条知识文档
     │
     ▼
② 构建 Prompt
   SystemMessage: "根据以下知识库内容回答：{context}"
   HumanMessage: "你们有什么课程？"
     │
     ▼
③ 调用 LLM
   llm.invoke(messages, config=RunnableConfig(callbacks=[StreamCollector]))
     │
     ▼
④ 流式输出
   on_llm_new_token → 逐 token 推送给前端
```

### Q12：主项目的 RAG 和 Demo 有什么区别？

```
Demo:                              主项目:
  内存向量库（3 条文档）              Milvus + ES 双路存储
  余弦相似度检索                     MixRetriever（向量 + BM25 混合）
  无 Rerank                         Reranker + 断崖截断 + 关键词保护
  无查询改写                         LLM 改写 + 规则兜底
  stuff 一次塞入 LLM                 max_content 限制 + 来源限流
  无知识图谱                         GraphRAG 增强（实体/关系/社区报告）
```

---

## 四、工程实践

### Q13：Demo 怎么做到每一步代码都可视化的？

核心是 `StepTracer` — 在每个关键操作处手动调用 `tracer.trace()` 埋点：

```python
tracer.trace(
    phase='build',                  # 阶段：build / execute
    title='添加条件边 — 意图路由',    # 步骤标题
    code='▶ graph.add_conditional_edges(...)',  # 代码片段
    explanation='条件边根据意图动态路由...',      # 教学说明
    component='LangGraph:conditional_edges',    # 组件标识
)
```

事件通过 `asyncio.Queue → SSE` 实时推送到前端，前端收集后支持步进/回放。

### Q14：怎么适配通义千问（非 OpenAI 模型）？

LangChain 的 `ChatOpenAI` 支持 `openai_api_base` 参数，天然兼容任何 OpenAI 接口格式的模型：

```python
llm = ChatOpenAI(
    openai_api_key='sk-xxx',
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    model='qwen-max',
)
```

不需要额外的适配包。但 LlamaIndex 的 `OpenAI` 类有模型名白名单，需要用 `OpenAILike`。

### Q15：从 Demo 到主项目，最大的架构差异是什么？

**动态 vs 静态**：

```
Demo: 工作流写死在 Python 代码里
  graph.add_node("intent", intent_node)
  graph.add_edge(START, "intent")
  → 每次都是同一个工作流

主项目: 从数据库 JSON 动态编译
  workflow_data = FlowDao.get_flow_by_id(workflow_id).data
  → JSON 里有 nodes 数组和 edges 数组
  → GraphEngine 遍历 JSON，动态 add_node / add_edge
  → 前端拖拽编辑 → 保存 JSON → 后端编译执行

这就是为什么主项目有 NodeFactory + 17 种节点类型：
  NodeFactory.instance_node(node_type="rag", ...)
  NodeFactory.instance_node(node_type="llm", ...)
  → 根据 JSON 中的 type 字段动态实例化
```

### Q16：LangGraph 在主项目中还解决了什么问题？

**1. 扇入等待**：多个并行分支汇聚到一个节点时，自动等待所有分支完成

```python
# 传入列表 = 等待 A、B、C 全部完成后才执行 D
graph.add_edge([A, B, C], D)
```

**2. 互斥分支检测**：自动判断两个分支是否来自同一个 Condition 的不同路径，如果是则不需要等待

**3. 循环节点**：通过 `SubGraphExecutor` 在循环体内嵌套子工作流

**4. 子工作流调用**：`WorkflowExecutorNode` 调用另一个工作流，实现工作流复用

### Q17：LangChain 和 LangGraph 在生产项目中分别承担什么角色？

```
LangGraph 承担（编排层）：
  ├── 工作流 DAG 构建和执行
  ├── 节点调度（拓扑排序）
  ├── 条件路由（意图识别 → 分支）
  ├── 状态管理（Redis 持久化）
  ├── 中断恢复（Human-in-the-Loop）
  └── 循环 / 子工作流

LangChain 承担（执行层）：
  ├── LLM 调用（ChatModel.invoke）
  ├── RAG 检索（VectorStore + Retriever）
  ├── Agent 智能体（ReAct / Function Call）
  ├── 流式输出（Callback → Redis → SSE）
  ├── 对话记忆（ConversationBufferWindowMemory）
  └── 工具调用（BaseTool / MCP 工具）
```

### Q18：如果让你从零搭一个类似的工作流系统，你会怎么设计？

> 1. **前端**用 ReactFlow 做拖拽编排，保存为 JSON（nodes + edges）
> 2. **后端**用 LangGraph StateGraph 动态编译 JSON 为可执行图
> 3. **节点体系**用工厂模式（NodeFactory），每种节点类型一个类，继承 BaseNode
> 4. **状态持久化**用 Redis（自定义 BaseCheckpointSaver）
> 5. **LLM 调用**用 LangChain 的 ChatModel + Callback 做流式输出
> 6. **RAG 检索**用 Milvus（向量）+ ES（BM25），双路混合检索
> 7. **事件推送**用 Redis 队列 + SSE/WebSocket
> 8. **异步执行**用线程池（ThreadPoolExecutor），API 层不阻塞

这基本就是主项目 ZFAPT 的架构。

### Q19：主项目用了 17 种节点，能列举几个典型的吗？

| 节点类型 | LangChain/LangGraph 用到的 API |
|---------|------------------------------|
| LLM 节点 | `ChatModel.invoke(messages, config)` + Callback 流式 |
| RAG 节点 | `VectorStore.search` + `PromptTemplate` + `ChatModel` |
| Agent 节点 | `AgentExecutor` + `BaseTool` + ReAct/Function Call |
| 意图识别 | `ChatModel.invoke` + JSON 解析 + `route_node()` |
| 条件分支 | `add_conditional_edges` + 表达式评估 |
| Input/Output | `interrupt_before` + `continue_run` |
| 报告节点 | `ChatModel` + docx/xlsx/pptx 模板渲染 |
| 代码节点 | Python 沙箱执行 |
| HTTP 节点 | 纯 HTTP 请求，不用 LangChain |

### Q20：这个 Demo 项目对你学习有什么帮助？

> 最大的帮助是**看到了 LangGraph 和 LangChain 的分工边界**。
>
> 之前看主项目代码时，分不清哪些是 LangGraph 的能力（图编排），哪些是 LangChain 的能力（模型调用）。通过这个 Demo 的逐步可视化：
>
> - **图构建阶段**（9 步）全是 LangGraph API：StateGraph → add_node → add_edge → conditional_edges → compile
> - **图执行阶段**（7 步）内部全是 LangChain API：Messages → invoke → Callback → VectorStore
>
> 两者通过节点函数 `run(state) -> state` 连接。理解了这个，再回去看主项目的 GraphEngine + 17 种节点就清晰多了。
