from collections.abc import Iterator
from typing import TypeVar

import httpx
from pydantic import BaseModel, JsonValue, StrictStr

from aadd.config import Settings
from aadd.external import ServiceError, call_service


Response = TypeVar('Response', bound=BaseModel)


def strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def replace_strings(value: JsonValue, replacements: Iterator[str]) -> JsonValue:
    if isinstance(value, str):
        return next(replacements)
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


class MaskedPart(BaseModel):
    id: StrictStr
    masked_text: StrictStr


class MaskResult(BaseModel):
    masked_parts: list[MaskedPart]
    mapping: list[dict[str, JsonValue]]


class UnmaskResult(BaseModel):
    restored_text: StrictStr


class Redactor:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def post(self, endpoint: str, payload: dict, schema: type[Response]) -> Response:
        async def operation():
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                response = await client.post(self.settings.redactor_url.rstrip('/') + endpoint, json=payload,
                                             headers=self.settings.redactor_headers)
                response.raise_for_status()
                return schema.model_validate(response.json())

        return await call_service(operation, self.settings, 'マスクAPIの処理')

    async def mask(self, texts: list[str]) -> MaskResult:
        parts = [{'id': str(index), 'text': text} for index, text in enumerate(texts)]
        payload = {'parts': parts, 'detection': self.settings.redactor_detection,
                   'mask_level': self.settings.redactor_mask_level}
        result = await self.post('/mask', payload, MaskResult)
        if [part.id for part in result.masked_parts] != [part['id'] for part in parts]:
            raise ServiceError('マスクAPIの応答に入力と対応しないパートがあります。')
        return result

    async def unmask(self, value: JsonValue, mapping: list[dict[str, JsonValue]]) -> JsonValue:
        restored = []
        for text in strings(value):
            result = await self.post('/unmask', {'text': text, 'mapping': mapping}, UnmaskResult)
            restored.append(result.restored_text)
        return replace_strings(value, iter(restored))
