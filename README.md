# LangChain + LangGraph 执行流程可视化教学 Demo

一个可交互的教学工具，通过可视化方式展示 LangChain 和 LangGraph 在实际项目中的协作流程。每一步代码执行实时高亮，让你直观看到 **LangGraph 管"图怎么走"、LangChain 管"节点怎么干"**。

## 快速启动

```bash
cd langchain_langgraph_demo

# 1. 创建虚拟环境（首次）
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 LLM（编辑项目根目录 .env）
# LLM_API_KEY=sk-xxx
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL=qwen-max

# 3. 启动服务
python backend/main.py

# 4. 打开浏览器
# http://localhost:8765
```

## 怎么提问？

Demo 内置了一个"智能客服"场景，支持 **三种意图路径**，每条路径展示不同的 LangChain / LangGraph 功能：

### 路径一：课程咨询（RAG 检索 + LLM 生成）

> 展示：VectorStore 向量检索 → PromptTemplate → ChatModel.invoke + Callback 流式输出

试试这些问题：
- `你们有什么课程？`
- `AI大模型课程多少钱？`
- `Python入门课什么时候上课？`
- `有没有数据分析相关的培训？`
- `LangChain课程在哪上课？`

### 路径二：报名（Human-in-the-Loop 中断/恢复）

> 展示：interrupt_before 暂停 → 前端表单输入 → continue_run 恢复执行

试试这些问题：
- `我想报名`
- `怎么报名AI课程？`
- `帮我注册一下`
- `我要参加Python培训`

输入后工作流会**暂停**，弹出姓名/手机号表单，填写提交后工作流**恢复执行**。

### 路径三：闲聊（直接 LLM 对话）

> 展示：最简单的 SystemMessage + HumanMessage → ChatModel.invoke

试试这些问题：
- `你好`
- `今天天气怎么样？`
- `给我讲个笑话`
- `你是谁？`

## 知识库内容

内置 3 条课程知识文档（内存向量库，启动时自动加载）：

| 文档 | 内容 |
|------|------|
| 课程总览 | 提供三门课程：**Python入门**（零基础）、**AI大模型应用开发**（有编程经验）、**数据分析实战**（数据岗位） |
| Python入门 | 周一到周五 9:00-12:00，共4周，**3000元**，A栋3楼301教室 |
| AI大模型 | 每周六 14:00-18:00，共8周，**5000元**，B栋5楼实验室。内容：LangChain、RAG、LangGraph、Agent |

## 页面说明

```
┌──────────────────┬────────────────┬──────────────────────┐
│ 左栏：工作流图     │ 中栏：对话区    │ 右栏：代码执行面板     │
│                  │               │                      │
│ 节点颜色含义：     │ - 输入提问     │ - 步骤标题            │
│ ⚫ 灰色 = 待执行   │ - AI 回复      │ - 代码片段（高亮行）   │
│ 🔴 红色 = 执行中   │ - 流式打字效果  │ - 输入/输出数据       │
│ 🟢 绿色 = 已完成   │ - 表单(中断时)  │ - 教学说明            │
│ 🟡 黄色 = 等待输入 │               │ - 组件标签：           │
│                  │               │   🔷蓝色=LangGraph    │
│                  │               │   🟢绿色=LangChain    │
├──────────────────┴────────────────┤                      │
│ 底部：步骤时间线（可点击跳转）       │ [上一步][下一步][自动] │
└────────────────────────────────┴──────────────────────┘
```

**操作方式**：
- 输入问题按回车 → 工作流开始执行，右侧逐步展示代码
- 点击时间线的任意节点 → 跳转到该步骤查看详情
- **上一步 / 下一步** → 手动步进浏览每一步
- **自动播放** → 1.5 秒间隔自动前进
- **重置** → 清空所有状态重新开始

## 覆盖的知识点

### LangGraph（蓝色标签）

| 步骤 | API | 说明 |
|------|-----|------|
| 创建图 | `StateGraph(TypedDict)` | 定义状态结构，创建图构建器 |
| 注册节点 | `graph.add_node(name, func)` | 将函数注册为图节点 |
| 固定边 | `graph.add_edge(A, B)` | A 执行完一定执行 B |
| 条件边 | `graph.add_conditional_edges(src, func, mapping)` | 动态路由 |
| 编译图 | `graph.compile(checkpointer, interrupt_before)` | 生成可执行图 |
| 执行图 | `graph.stream(input, config)` | 按拓扑序逐节点执行 |
| 中断 | `interrupt_before=["register"]` | 人机交互暂停 |
| 恢复 | `graph.stream(None, config)` | 从 checkpoint 恢复 |

### LangChain（绿色标签）

| 步骤 | API | 说明 |
|------|-----|------|
| 构建消息 | `SystemMessage` / `HumanMessage` | LLM 交互的消息协议 |
| 调用模型 | `llm.invoke(messages, config)` | 核心调用方法 |
| 流式回调 | `BaseCallbackHandler.on_llm_new_token` | 拦截每个生成的 token |
| 配置注入 | `RunnableConfig(callbacks=[...])` | 将回调注入调用链 |
| 向量检索 | `vectorstore.similarity_search(query, k)` | RAG 检索核心 |
| 提示词模板 | `ChatPromptTemplate` | 将检索结果嵌入 Prompt |

## 项目结构

```
langchain_langgraph_demo/
├── .env                     # LLM 配置（项目根目录）
├── requirements.txt         # Python 依赖
├── README.md
├── backend/
│   ├── main.py              # FastAPI 入口，4 个 API
│   ├── config.py            # 读取 .env 配置
│   ├── step_tracer.py       # 步骤追踪器，SSE 推送
│   ├── workflow_engine.py   # 埋点版 LangGraph 工作流引擎
│   ├── nodes.py             # 4 个节点（LangChain 调用）
│   └── knowledge_base.py   # 内存向量知识库
├── static/
│   └── index.html           # 前端页面（三栏可视化）
└── venv/                    # Python 虚拟环境
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/run` | 启动工作流，SSE 流式返回步骤事件 |
| POST | `/api/continue` | 恢复中断的工作流（提交表单数据） |
| POST | `/api/reset` | 重置工作流状态 |
| GET | `/api/steps` | 获取所有已执行的步骤历史 |
