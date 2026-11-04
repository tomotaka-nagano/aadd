import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aadd.config import Settings


Result = TypeVar('Result')


class ServiceError(ValueError):
    pass


async def call_service(operation: Callable[[], Awaitable[Result]], settings: Settings, label: str) -> Result:
    for attempt in range(settings.retry_count + 1):
        try:
            async with asyncio.timeout(settings.timeout_seconds):
                return await operation()
        except ServiceError:
            raise
        except Exception as error:
            if attempt == settings.retry_count:
                raise ServiceError(f'{label}に失敗しました（{type(error).__name__}）。設定と接続を確認してください。') from error
            await asyncio.sleep(min(2 ** attempt, 4))
    raise ServiceError(f'{label}に失敗しました。')
