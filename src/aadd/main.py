import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aadd.api import router
from aadd.config import Settings, load_settings
from aadd.storage import Conflict, Store
from aadd.workflow import Workflow


STATIC = Path(__file__).parent / 'static'


class BodyLimit:
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > self.limit:
                raise HTTPException(413, 'リクエストサイズが上限を超えています。')
            return message

        await self.app(scope, bounded_receive, send)


def lifespan_for(settings, ai, kb):
    @asynccontextmanager
    async def lifespan(application):
        store = await asyncio.to_thread(Store, settings.data_dir)
        await asyncio.to_thread(store.interrupt)
        application.state.store = store
        application.state.workflow = Workflow(store, settings, ai, kb)
        yield
        await application.state.workflow.close()

    return lifespan


async def missing(request: Request, error: KeyError):
    return JSONResponse({'detail': str(error.args[0])}, status_code=404)


async def conflict(request: Request, error: Conflict):
    return JSONResponse({'detail': str(error)}, status_code=409)


async def invalid(request: Request, error: ValueError):
    return JSONResponse({'detail': str(error)}, status_code=422)


async def same_origin(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin and request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        if urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': '異なるオリジンからの操作は許可されていません。'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
    )
    response.headers['Cache-Control'] = 'no-store'
    return response


def create_app(settings: Settings | None = None, *, ai=None, kb=None) -> FastAPI:
    settings = settings or load_settings()
    application = FastAPI(title='AADD', lifespan=lifespan_for(settings, ai, kb))
    application.state.settings = settings
    application.add_exception_handler(KeyError, missing)
    application.add_exception_handler(Conflict, conflict)
    application.add_exception_handler(ValueError, invalid)
    application.include_router(router)
    application.mount('/static', StaticFiles(directory=STATIC, check_dir=False), name='static')
    application.add_api_route('/', lambda: FileResponse(STATIC / 'index.html'), include_in_schema=False)
    limit = settings.max_files * settings.max_file_bytes + settings.max_input_chars * 4 + 1024 * 1024
    application.add_middleware(BodyLimit, limit=limit)
    application.middleware('http')(same_origin)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    return application


app = create_app()
