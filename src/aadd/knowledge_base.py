import asyncio
import base64
import binascii
import hashlib
import json
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import httpx
from markitdown import MarkItDownException, StreamInfo
from markitdown.converters import XlsxConverter
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from aadd.attachments import validate_archive
from aadd.config import Settings
from aadd.external import ServiceError, call_service
from aadd.models import Candidate, Change, Snapshot


class SearchResponse(BaseModel):
    documents: list[Candidate]


def tool_data(result: CallToolResult) -> dict[str, Any]:
    if result.isError:
        raise ServiceError('ナレッジベースのMCPツールがエラーを返しました。KBのログを確認してください。')
    data = result.structuredContent
    if data is None and len(result.content) == 1 and isinstance(result.content[0], TextContent):
        data = json.loads(result.content[0].text)
    if not isinstance(data, dict) or data.get('error'):
        raise ServiceError('ナレッジベースのMCP応答が不正、または処理に失敗しました。KBのログを確認してください。')
    return data


def decode_workbook(content: str, max_bytes: int) -> bytes:
    if len(content) > 4 * ((max_bytes + 2) // 3):
        raise ServiceError('仕様書がファイルサイズ上限を超えています。')
    try:
        workbook = base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ServiceError('仕様書の Base64 データが不正です。') from error
    if not workbook or len(workbook) > max_bytes:
        raise ServiceError('仕様書が空、またはファイルサイズ上限超過です。')
    return workbook


def workbook_markdown(workbook: bytes, settings: Settings) -> str:
    try:
        validate_archive(workbook, '.xlsx', settings)
        with BytesIO(workbook) as stream:
            return XlsxConverter().convert(stream, StreamInfo(extension='.xlsx')).markdown
    except (BadZipFile, KeyError, MarkItDownException, ParseError, TypeError, ValueError) as error:
        raise ServiceError('Excel 仕様書を Markdown に変換できませんでした。ファイル形式やサイズを確認してください。') from error


class KnowledgeBase:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def call_tool(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        async def operation():
            url = httpx.URL(self.settings.kb_url)
            url = url.copy_with(path='/mcp') if url.path in {'', '/'} else url
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport,
                                         headers=self.settings.kb_headers) as client:
                async with streamable_http_client(str(url), http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await session.call_tool(name, payload)

        result = await call_service(operation, self.settings, 'ナレッジベースの処理')
        try:
            return tool_data(result)
        except ServiceError:
            raise
        except (ValueError, TypeError) as error:
            raise ServiceError('ナレッジベースのMCP応答を読み取れません。KBのログを確認してください。') from error

    async def search(self, change: Change) -> list[Candidate]:
        return await self.search_query('\n'.join(filter(None, [change.summary, change.before, change.after])))

    async def search_query(self, query: str) -> list[Candidate]:
        payload = {'query': query, 'limit': self.settings.candidate_limit, 'mode': self.settings.search_mode,
                   'scope_paths': self.settings.scope_paths or None,
                   'exclude_paths': self.settings.exclude_paths or None}
        result = SearchResponse.model_validate(await self.call_tool('search_documents', payload))
        documents = {}
        for item in result.documents:
            if PurePosixPath(item.file_path).suffix.lower() == '.xlsx':
                if item.document_id not in documents or item.max_score > documents[item.document_id].max_score:
                    documents[item.document_id] = item
        return sorted(documents.values(), key=lambda item: (-item.max_score, item.document_id))

    async def download(self, candidate: Candidate) -> Snapshot:
        payload = {'document_id': candidate.document_id, 'encoding': 'base64', 'include_images': False}
        data = await self.call_tool('download_document', payload)
        if data.get('status') != 'success':
            raise ServiceError('仕様書の取得に失敗しました。KB上の文書を確認してください。')
        if data.get('encoding') != 'base64':
            raise ServiceError('仕様書のエンコーディングが不正です。')
        content = data.get('content')
        if not isinstance(content, str):
            raise ServiceError('仕様書の本文が不正です。')
        workbook = decode_workbook(content, self.settings.max_file_bytes)
        text = await asyncio.to_thread(workbook_markdown, workbook, self.settings)
        if not text.strip() or len(text) > self.settings.max_document_chars:
            raise ServiceError('仕様書が空、または文字数上限超過です。')
        return Snapshot(document_id=candidate.document_id, file_path=candidate.file_path, text=text,
                        sha256=hashlib.sha256(text.encode()).hexdigest())
