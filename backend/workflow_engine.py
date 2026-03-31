# AI_GENERATE_START
"""
埋点版工作流引擎
在每个关键操作处调用 tracer.trace() 记录步骤
"""
import operator
from typing import Annotated, Any, Optional

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from backend.nodes import (
    intent_node, route_intent, rag_node,
    register_node, chat_node,
)
from backend.step_tracer import StepTracer


class WorkflowState(TypedDict):
    """工作流状态定义"""
    query: str
    intent: str
    answer: str
    source_docs: list
    user_input: dict
    _dummy: Annotated[list, operator.add]


class WorkflowEngine:
    """埋点版工作流引擎"""

    def __init__(self):
        self.tracer = StepTracer()
        self.graph = None
        self.config = None
        self._checkpointer = MemorySaver()
        self._thread_id = 'demo_thread'

    def build_and_run(self, query: str):
        """构建图 + 执行图"""
        self.tracer.reset()
        self._build_graph()
        self._run_graph(query)

    def continue_run(self, user_input: dict):
        """恢复中断的图执行"""
        if not self.graph:
            return

        self.tracer.trace(
            phase='execute', title='恢复执行 — continue_run(None)',
            code=(
                '# 用户提交表单后，传入 None 从 checkpoint 恢复\n'
                '▶ for event in graph.stream(None, config=config):\n'
                '    pass  # 从 interrupt 处继续执行'
            ),
            explanation=(
                'LangGraph 的 interrupt 恢复机制：\n'
                'graph.stream(None, config) 传入 None 表示从上次中断处继续。\n'
                '图的状态已经通过 checkpointer 保存，恢复后接着执行下一个节点。'
            ),
            component='LangGraph:continue',
        )

        # 更新 state 中的用户输入
        self.graph.update_state(self.config, {'user_input': user_input})

        # 恢复执行
        for event in self.graph.stream(None, config=self.config):
            pass

        self.tracer.trace(
            phase='execute', title='到达 END 节点 — 工作流结束',
            code='→ END',
            node_id='end', node_status='completed',
            explanation='所有节点执行完毕，图到达 END，工作流结束。',
            component='LangGraph:END',
        )
        self.tracer.finish()

    # ================================================================
    # 图构建阶段
    # ================================================================
    def _build_graph(self):
        """构建 LangGraph StateGraph，每一步都埋点"""

        # Step: 创建 StateGraph
        builder = StateGraph(WorkflowState)
        self.tracer.trace(
            phase='build', title='创建 StateGraph',
            code=(
                'from langgraph.graph import StateGraph\n'
                'from typing_extensions import TypedDict\n'
                '\n'
                'class WorkflowState(TypedDict):\n'
                '    query: str\n'
                '    intent: str\n'
                '    answer: str\n'
                '\n'
                '▶ graph_builder = StateGraph(WorkflowState)'
            ),
            explanation=(
                'StateGraph 是 LangGraph 的核心类。\n'
                '传入一个 TypedDict 定义状态结构，图中所有节点共享这个状态。\n'
                '每个节点读取 state 中的数据，处理后更新 state。'
            ),
            component='LangGraph:StateGraph',
        )

        # Step: 注册节点
        tracer = self.tracer

        def _intent(state):
            return intent_node(state, tracer)

        def _rag(state):
            return rag_node(state, tracer)

        def _register(state):
            return register_node(state, tracer)

        def _chat(state):
            return chat_node(state, tracer)

        # 注册意图识别节点
        builder.add_node('intent', _intent)
        self.tracer.trace(
            phase='build', title='注册节点 — 意图识别',
            code='▶ graph_builder.add_node("intent", intent_node)',
            node_id='intent', node_status='pending',
            explanation=(
                'add_node(name, func) 将一个函数注册为图的节点。\n'
                '函数签名：func(state: dict) -> dict\n'
                '接收当前状态，返回更新后的状态。'
            ),
            component='LangGraph:add_node',
        )

        # 注册 RAG 节点
        builder.add_node('rag', _rag)
        self.tracer.trace(
            phase='build', title='注册节点 — RAG 检索+生成',
            code='▶ graph_builder.add_node("rag", rag_node)',
            node_id='rag', node_status='pending',
            explanation='RAG 节点内部使用 LangChain 的 VectorStore + ChatModel 完成检索增强生成。',
            component='LangGraph:add_node',
        )

        # 注册报名节点
        builder.add_node('register', _register)
        self.tracer.trace(
            phase='build', title='注册节点 — 报名流程',
            code='▶ graph_builder.add_node("register", register_node)',
            node_id='register', node_status='pending',
            explanation='报名节点使用 interrupt_before 实现人机交互中断。',
            component='LangGraph:add_node',
        )

        # 注册闲聊节点
        builder.add_node('chat', _chat)
        self.tracer.trace(
            phase='build', title='注册节点 — 闲聊',
            code='▶ graph_builder.add_node("chat", chat_node)',
            node_id='chat', node_status='pending',
            explanation='闲聊节点直接调用 LangChain ChatModel 生成回复。',
            component='LangGraph:add_node',
        )

        # Step: 添加起始边
        builder.add_edge(START, 'intent')
        self.tracer.trace(
            phase='build', title='添加起始边 START → 意图识别',
            code=(
                'from langgraph.constants import START, END\n'
                '▶ graph_builder.add_edge(START, "intent")'
            ),
            explanation=(
                'add_edge(A, B) 添加固定边：A 执行完一定执行 B。\n'
                'START 是 LangGraph 的特殊常量，表示图的入口。'
            ),
            component='LangGraph:add_edge',
        )

        # Step: 添加条件边
        builder.add_conditional_edges(
            'intent',
            route_intent,
            {'rag': 'rag', 'register': 'register', 'chat': 'chat'},
        )
        self.tracer.trace(
            phase='build', title='添加条件边 — 意图路由',
            code=(
                'def route_intent(state: dict) -> str:\n'
                '    """路由函数直接返回目标节点 ID"""\n'
                '    intent = state["intent"]\n'
                '    return {"课程咨询": "rag", "报名": "register", "闲聊": "chat"}[intent]\n'
                '\n'
                '▶ graph_builder.add_conditional_edges(\n'
                '    "intent",          # 源节点\n'
                '    route_intent,       # 路由函数（返回节点ID）\n'
                '    {"rag": "rag", "register": "register", "chat": "chat"}\n'
                ')'
            ),
            explanation=(
                'add_conditional_edges 是 LangGraph 实现动态路由的核心 API。\n'
                '源节点执行完后，调用路由函数获取返回值，\n'
                '在 mapping 字典中找到对应的目标节点，继续执行。\n'
                '这就是主项目中意图识别 → 多分支的实现原理。'
            ),
            component='LangGraph:conditional_edges',
        )

        # Step: 添加终止边
        builder.add_edge('rag', END)
        builder.add_edge('register', END)
        builder.add_edge('chat', END)
        self.tracer.trace(
            phase='build', title='添加终止边 → END',
            code=(
                '▶ graph_builder.add_edge("rag", END)\n'
                '  graph_builder.add_edge("register", END)\n'
                '  graph_builder.add_edge("chat", END)'
            ),
            explanation='END 是 LangGraph 的特殊常量，表示图的出口。所有分支最终都汇聚到 END。',
            component='LangGraph:add_edge',
        )

        # Step: 编译图
        self.graph = builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=['register'],
        )
        self.tracer.trace(
            phase='build', title='编译图 — compile',
            code=(
                'from langgraph.checkpoint.memory import MemorySaver\n'
                '\n'
                '▶ graph = graph_builder.compile(\n'
                '    checkpointer=MemorySaver(),       # 状态持久化\n'
                '    interrupt_before=["register"]      # 报名节点前中断\n'
                ')'
            ),
            explanation=(
                'compile() 将图构建器编译为可执行的图。\n'
                '• checkpointer：每执行一步自动保存状态，支持中断后恢复。\n'
                '  主项目使用自定义的 RedisSaver，这里用内存版 MemorySaver。\n'
                '• interrupt_before：执行到这些节点前自动暂停（Human-in-the-Loop）。\n'
                '  暂停后通过 graph.stream(None, config) 恢复。'
            ),
            component='LangGraph:compile',
        )

    # ================================================================
    # 图执行阶段
    # ================================================================
    def _run_graph(self, query: str):
        """执行图"""
        self.config = {'configurable': {'thread_id': self._thread_id}}

        input_data = {
            'query': query,
            'intent': '',
            'answer': '',
            'source_docs': [],
            'user_input': {},
            '_dummy': [],
        }

        # Step: 启动执行
        self.tracer.trace(
            phase='execute', title='启动图执行 — graph.stream()',
            code=(
                'input_data = {"query": "' + query[:30] + '..."}\n'
                'config = {"configurable": {"thread_id": "demo"}}\n'
                '\n'
                '▶ for event in graph.stream(input_data, config=config):\n'
                '    pass  # LangGraph 按拓扑序逐个执行节点'
            ),
            input_data={'query': query},
            explanation=(
                'graph.stream() 是 LangGraph 执行图的核心方法。\n'
                '传入初始状态和配置，按拓扑顺序逐个执行节点。\n'
                '每执行完一个节点 yield 一次结果。\n'
                '如果遇到 interrupt_before 节点，执行会自动暂停。'
            ),
            component='LangGraph:stream',
        )

        # 实际执行
        interrupted = False
        for event in self.graph.stream(input_data, config=self.config):
            pass

        # 检查是否被中断
        snapshot = self.graph.get_state(self.config)
        if snapshot.next:
            # 被 interrupt 了
            interrupted = True
            self.tracer.trace(
                phase='execute', title='图执行中断 — interrupt_before',
                code=(
                    '# graph.get_state(config) 获取当前快照\n'
                    'snapshot = graph.get_state(config)\n'
                    '▶ snapshot.next  # → ("register",)\n'
                    '\n'
                    '# next 不为空说明图被中断，等待用户输入'
                ),
                node_id='register', node_status='waiting',
                output_data={'next_nodes': list(snapshot.next), 'status': 'interrupted'},
                explanation=(
                    'LangGraph 的 interrupt_before 机制：\n'
                    '执行到 register 节点前自动暂停。\n'
                    'get_state().next 返回下一步要执行的节点列表。\n'
                    '如果不为空，说明图被中断了。\n'
                    '前端此时应展示输入表单，用户提交后调用 continue_run。'
                ),
                component='LangGraph:interrupt',
            )
            self.tracer.finish()
            return

        # 正常结束
        self.tracer.trace(
            phase='execute', title='到达 END 节点 — 工作流结束',
            code='→ END',
            node_id='end', node_status='completed',
            explanation='所有节点执行完毕，图到达 END，工作流结束。',
            component='LangGraph:END',
        )

        # 路由步骤（在意图识别后补充一条路由记录）
        self.tracer.finish()


# 全局引擎实例
engine = WorkflowEngine()
# AI_GENERATE_END
