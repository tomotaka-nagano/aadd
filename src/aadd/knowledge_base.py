import base64
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

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
            if PurePosixPath(item.file_path).suffix.lower() == '.md':
                if item.document_id not in documents or item.max_score > documents[item.document_id].max_score:
                    documents[item.document_id] = item
        return sorted(documents.values(), key=lambda item: (-item.max_score, item.document_id))

    async def download(self, candidate: Candidate) -> Snapshot:
        payload = {'document_id': candidate.document_id, 'encoding': 'text', 'include_images': False}
        data = await self.call_tool('download_document', payload)
        if data.get('status') != 'success':
            raise ServiceError('仕様書の取得に失敗しました。KB上の文書を確認してください。')
        if data.get('encoding') not in {'text', 'base64'}:
            raise ServiceError('仕様書のエンコーディングが不正です。')
        content = data.get('content')
        if not isinstance(content, str):
            raise ServiceError('仕様書の本文が不正です。')
        text = base64.b64decode(content, validate=True).decode('utf-8-sig') if data['encoding'] == 'base64' else content
        if not text.strip() or len(text) > self.settings.max_document_chars:
            raise ServiceError('仕様書が空、または文字数上限超過です。')
        return Snapshot(document_id=candidate.document_id, file_path=candidate.file_path, text=text,
                        sha256=hashlib.sha256(text.encode()).hexdigest())
