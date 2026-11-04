import base64
import hashlib
from pathlib import PurePosixPath

import httpx
from pydantic import BaseModel

from aadd.config import Settings
from aadd.external import ServiceError, call_service
from aadd.models import Candidate, Change, Snapshot


class SearchResponse(BaseModel):
    documents: list[Candidate]


class KnowledgeBase:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.transport = transport

    async def post(self, endpoint: str, payload: dict) -> dict:
        async def operation():
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                response = await client.post(self.settings.kb_url.rstrip('/') + endpoint, json=payload,
                                             headers=self.settings.kb_headers)
                response.raise_for_status()
                return response.json()

        return await call_service(operation, self.settings, 'ナレッジベースの処理')

    async def search(self, change: Change) -> list[Candidate]:
        payload = {'query': '\n'.join(filter(None, [change.summary, change.before, change.after])),
                   'limit': self.settings.candidate_limit, 'mode': self.settings.search_mode,
                   'scope_paths': self.settings.scope_paths or None,
                   'exclude_paths': self.settings.exclude_paths or None}
        result = SearchResponse.model_validate(await self.post('/api/search_documents', payload))
        documents = {}
        for item in result.documents:
            if PurePosixPath(item.file_path).suffix.lower() == '.md':
                if item.document_id not in documents or item.max_score > documents[item.document_id].max_score:
                    documents[item.document_id] = item
        return sorted(documents.values(), key=lambda item: (-item.max_score, item.document_id))

    async def download(self, candidate: Candidate) -> Snapshot:
        payload = {'document_id': candidate.document_id, 'encoding': 'text', 'include_images': False}
        data = await self.post('/api/download_document', payload)
        if data.get('status') != 'success':
            raise ServiceError('仕様書の取得に失敗しました。KB上の文書を確認してください。')
        if data.get('encoding') not in {'text', 'base64'}:
            raise ServiceError('仕様書のエンコーディングが不正です。')
        content = data['content']
        text = base64.b64decode(content, validate=True).decode('utf-8-sig') if data['encoding'] == 'base64' else content
        if not text.strip() or len(text) > self.settings.max_document_chars:
            raise ServiceError('仕様書が空、または文字数上限超過です。')
        return Snapshot(document_id=candidate.document_id, file_path=candidate.file_path, text=text,
                        sha256=hashlib.sha256(text.encode()).hexdigest())
