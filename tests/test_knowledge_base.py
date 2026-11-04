import base64

import httpx
import pytest

from aadd.config import Settings
from aadd.knowledge_base import KnowledgeBase
from aadd.models import Candidate, Change


async def test_search_deduplicates_and_filters():
    def handler(request):
        assert request.url.path == '/api/search_documents'
        item = {'document_id': 'a', 'file_path': 'spec.md', 'title': 'Spec', 'max_score': 0.8}
        return httpx.Response(200, json={'documents': [item, item, {**item, 'document_id': 'b', 'file_path': 'x.pdf'}]})

    kb = KnowledgeBase(Settings(), httpx.MockTransport(handler))
    assert len(await kb.search(Change(summary='変更'))) == 1


@pytest.mark.parametrize('encoding', ['text', 'base64'])
async def test_download_original_markdown(encoding):
    text = '# 仕様\n変更前\n'

    def handler(request):
        assert request.url.path == '/api/download_document'
        content = text if encoding == 'text' else base64.b64encode(text.encode()).decode()
        return httpx.Response(200, json={'status': 'success', 'content': content, 'encoding': encoding})

    kb = KnowledgeBase(Settings(), httpx.MockTransport(handler))
    snapshot = await kb.download(Candidate(document_id='a', file_path='spec.md', title='Spec', max_score=0.8))
    assert snapshot.text == text
    assert len(snapshot.sha256) == 64


async def test_failed_download():
    kb = KnowledgeBase(Settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={'status': 'error'})))
    with pytest.raises(ValueError):
        await kb.download(Candidate(document_id='a', file_path='a.md', title='A', max_score=1))
