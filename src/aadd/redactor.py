from collections.abc import Iterator
import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, Field, JsonValue, StrictStr

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


def merge_masks(results: list[MaskResult]) -> list[MaskResult]:
    used = {entry.get('placeholder') for result in results for entry in result.mapping
            if isinstance(entry.get('placeholder'), str)}
    by_label: dict[str, str] = {}
    by_original: dict[str, str] = {}
    merged = []
    for result in results:
        labels: dict[str, str] = {}
        mapping = []
        for entry in result.mapping:
            label, original = entry.get('placeholder'), entry.get('canonical')
            surfaces = entry.get('surfaces')
            if not original and isinstance(surfaces, list) and surfaces:
                original = surfaces[0]
            if not isinstance(label, str) or not label or not isinstance(original, str) or not original:
                raise ServiceError('マスクAPIの対応表が不正です。')
            target = by_original.get(original, label)
            if target in by_label and by_label[target] != original:
                index = len(by_label) + 1
                target = f'[MASK_{index}]'
                while target in used or target in by_label or any(
                    target in part.masked_text for batch in results for part in batch.masked_parts
                ):
                    index += 1
                    target = f'[MASK_{index}]'
            if label in labels and labels[label] != target:
                raise ServiceError('マスクAPIの対応表内で伏せ字が衝突しています。')
            labels[label] = target
            if target not in by_label:
                mapping.append({**entry, 'placeholder': target})
                by_label[target] = original
                by_original[original] = target
        pattern = re.compile('|'.join(map(re.escape, sorted(labels, key=len, reverse=True)))) if labels else None
        parts = [MaskedPart(id=part.id, masked_text=pattern.sub(lambda m: labels[m[0]], part.masked_text))
                 for part in result.masked_parts] if pattern else result.masked_parts
        merged.append(MaskResult(masked_parts=parts, mapping=mapping))
    return merged


class UnmaskResult(BaseModel):
    restored_text: StrictStr


class DocumentReference(BaseModel):
    content_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


class ConvertedDocument(DocumentReference):
    chunks: list[StrictStr]


class Redactor:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def request(self, method: str, endpoint: str, schema: type[Response], *, payload: dict | None = None,
                      files: dict[str, tuple[str, bytes, str]] | None = None) -> Response:
        async def operation():
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                url = self.settings.redactor_url.rstrip('/') + endpoint
                headers = httpx.Headers(self.settings.redactor_headers)
                if files is None:
                    response = await client.request(method, url, json=payload, headers=headers)
                else:
                    headers.pop('content-type', None)
                    form = {'manifest': json.dumps(payload)} if payload is not None else None
                    response = await client.request(method, url, data=form, files=files, headers=headers)
                response.raise_for_status()
                return schema.model_validate(response.json())

        return await call_service(operation, self.settings, 'マスクAPIの処理')

    async def convert_workbook(self, filename: str, workbook: bytes) -> ConvertedDocument:
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        reference = await self.request('POST', '/documents', DocumentReference,
                                       files={'file': (filename, workbook, mime_type)})
        document = await self.request('GET', f'/documents/{reference.content_hash}', ConvertedDocument)
        if document.content_hash != reference.content_hash:
            raise ServiceError('マスクAPIから取得した変換結果の文書IDが一致しません。')
        return document

    async def mask(self, texts: list[str]) -> MaskResult:
        parts: list[dict[str, JsonValue]] = [{'id': str(index), 'text': text} for index, text in enumerate(texts)]
        return await self.mask_parts(parts)

    async def mask_markdown(self, text: str, *, filename: str) -> MaskResult:
        parts: list[dict[str, JsonValue]] = [{'id': '0', 'file': {'filename': filename}}]
        return await self.mask_parts(parts, files={'0': (filename, text.encode('utf-8'), 'text/markdown')})

    async def mask_document_reference(self, content_hash: str) -> MaskResult:
        return await self.mask_parts([{'id': '0', 'content_hash': content_hash}])

    async def mask_parts(self, parts: list[dict[str, JsonValue]], *,
                         files: dict[str, tuple[str, bytes, str]] | None = None) -> MaskResult:
        payload = {'parts': parts, 'detection': self.settings.redactor_detection,
                   'mask_level': self.settings.redactor_mask_level}
        result = await self.request('POST', '/mask', MaskResult, payload=payload, files=files)
        if [part.id for part in result.masked_parts] != [part['id'] for part in parts]:
            raise ServiceError('マスクAPIの応答に入力と対応しないパートがあります。')
        return result

    async def unmask(self, value: JsonValue, mapping: list[dict[str, JsonValue]]) -> JsonValue:
        restored = []
        for text in strings(value):
            result = await self.request('POST', '/unmask', UnmaskResult, payload={'text': text, 'mapping': mapping})
            restored.append(result.restored_text)
        return replace_strings(value, iter(restored))
