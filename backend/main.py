# AI_GENERATE_START
"""
FastAPI 入口
"""
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

# 将 demo 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.knowledge_base import knowledge_base
from backend.workflow_engine import engine


@asynccontextmanager
async def lifespan(app):
    """启动时初始化知识库"""
    knowledge_base.initialize()
    yield


app = FastAPI(title='LangChain + LangGraph 可视化教学 Demo', lifespan=lifespan)

# 挂载静态文件
static_dir = Path(__file__).resolve().parent.parent / 'static'
app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')


@app.get('/')
async def index():
    """主页面"""
    return FileResponse(str(static_dir / 'index.html'))


@app.post('/api/run')
async def run_workflow(query: str = Body(..., embed=True)):
    """启动工作流，返回 SSE 事件流"""
    engine.tracer.reset()

    # 在后台线程执行工作流（因为 LLM 调用是同步阻塞的）
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_run_in_thread(query))

    # 立即返回 SSE 流
    return StreamingResponse(
        engine.tracer.get_events(),
        media_type='text/event-stream',
    )


async def _run_in_thread(query: str):
    """在线程池中执行工作流"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.build_and_run, query)


@app.post('/api/continue')
async def continue_workflow(user_input: dict = Body(..., embed=True)):
    """恢复中断的工作流"""

    async def sse_generator():
        # 在后台线程恢复执行
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(
            loop.run_in_executor(None, engine.continue_run, user_input)
        )
        # 返回新产生的事件
        async for event in engine.tracer.get_events():
            yield event

    return StreamingResponse(
        sse_generator(),
        media_type='text/event-stream',
    )


@app.post('/api/reset')
async def reset_workflow():
    """重置工作流状态"""
    engine.tracer.reset()
    engine.graph = None
    engine.config = None
    # 重新创建引擎的 checkpointer
    from langgraph.checkpoint.memory import MemorySaver
    engine._checkpointer = MemorySaver()
    return JSONResponse({'status': 'ok'})


@app.get('/api/steps')
async def get_steps():
    """获取所有已执行的步骤"""
    return JSONResponse(engine.tracer.get_history())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8765)
# AI_GENERATE_END
