import asyncio
import base64
import hashlib
import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from aadd.config import Settings
from aadd.external import ServiceError
from aadd.knowledge_base import KnowledgeBase
from aadd.models import Candidate, Change


CANDIDATE = Candidate(document_id='a', file_path='spec.md', title='Spec', max_score=0.8)


def protocol_transport(data, *, structured=True, is_error=False):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == 'GET':
            return httpx.Response(405)
        if request.method == 'DELETE':
            return httpx.Response(200)
        message = json.loads(request.content)
        if 'id' not in message:
            return httpx.Response(202)
        if message['method'] == 'initialize':
            result = {'protocolVersion': message['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'kb-mcp-test', 'version': '1'}}
        elif message['method'] == 'tools/list':
            result = {'tools': []}
        else:
            assert message['method'] == 'tools/call'
            result = {'content': [{'type': 'text', 'text': json.dumps(data)}], 'isError': is_error}
            if structured:
                result['structuredContent'] = data
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': result},
                              headers={'mcp-session-id': 'test-session'})

    return httpx.MockTransport(handler), requests


@pytest.mark.parametrize('url', ['http://127.0.0.1:8000', 'http://127.0.0.1:8000/',
                                 'http://127.0.0.1:8000/mcp', 'http://127.0.0.1:8000/proxy/mcp'])
@pytest.mark.parametrize('structured', [True, False])
async def test_search_uses_mcp_and_preserves_filters(url, structured):
    item = CANDIDATE.model_dump()
    transport, requests = protocol_transport({'documents': [
        item, {**item, 'max_score': 0.9}, {**item, 'document_id': 'b', 'file_path': 'x.pdf'},
    ]}, structured=structured)
    settings = Settings(kb_url=url, kb_headers={'Authorization': 'Bearer test'}, scope_paths=['specs/'],
                        exclude_paths=['specs/old/'], search_mode='semantic', candidate_limit=3)
    result = await KnowledgeBase(settings, transport).search(Change(summary='変更', before='前', after='後'))
    assert len(result) == 1
    assert result[0].max_score == 0.9
    assert {request.url.path for request in requests} == {'/proxy/mcp' if '/proxy/' in url else '/mcp'}
    assert all(request.headers['Authorization'] == 'Bearer test' for request in requests)
    messages = [json.loads(request.content) for request in requests if request.method == 'POST']
    assert messages[0]['method'] == 'initialize'
    assert messages[1]['method'] == 'notifications/initialized'
    call = next(message for message in messages if message['method'] == 'tools/call')
    assert call['params'] == {'name': 'search_documents', 'arguments': {
        'query': '変更\n前\n後', 'limit': 3, 'mode': 'semantic',
        'scope_paths': ['specs/'], 'exclude_paths': ['specs/old/'],
    }}
    assert all(request.headers['mcp-session-id'] == 'test-session' for request in requests[1:])
    assert requests[-1].method == 'DELETE'


@pytest.mark.parametrize('encoding', ['text', 'base64'])
async def test_download_original_markdown(encoding):
    text = '# 仕様\n変更前\n'
    content = text if encoding == 'text' else base64.b64encode(text.encode()).decode()
    transport, requests = protocol_transport({'status': 'success', 'content': content, 'encoding': encoding})
    snapshot = await KnowledgeBase(Settings(), transport).download(CANDIDATE)
    assert snapshot.text == text
    assert snapshot.sha256 == hashlib.sha256(text.encode()).hexdigest()
    messages = [json.loads(request.content) for request in requests if request.method == 'POST']
    call = next(message for message in messages if message['method'] == 'tools/call')
    assert call['params'] == {'name': 'download_document', 'arguments': {
        'document_id': 'a', 'encoding': 'text', 'include_images': False,
    }}


@pytest.mark.parametrize('data, is_error', [
    ({'error': 'search failed'}, False), ({'documents': []}, True),
    ({'status': 'not_found'}, False), ({'status': 'error'}, False),
])
async def test_failed_download(data, is_error):
    transport, _ = protocol_transport(data, is_error=is_error)
    with pytest.raises(ServiceError):
        await KnowledgeBase(Settings(retry_count=0), transport).download(CANDIDATE)


@pytest.mark.parametrize('data', [{'error': 'search failed'}, ['not an object']])
async def test_failed_search_is_not_an_empty_success(data):
    transport, _ = protocol_transport(data, structured=False)
    with pytest.raises(ServiceError):
        await KnowledgeBase(Settings(retry_count=0), transport).search(Change(summary='変更'))


async def test_connection_failure_is_retried_with_a_new_session():
    transport, requests = protocol_transport({'documents': []})
    attempts = []

    async def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503)
        return await transport.handle_async_request(request)

    kb = KnowledgeBase(Settings(retry_count=1), httpx.MockTransport(handler))
    assert await kb.search(Change(summary='変更')) == []
    assert len([request for request in attempts if request.method == 'POST'
                and json.loads(request.content)['method'] == 'initialize']) == 2
    assert requests[-1].method == 'DELETE'


async def test_timeout_is_an_explicit_failure():
    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(503)

    kb = KnowledgeBase(Settings(retry_count=0, timeout_seconds=0.05), httpx.MockTransport(handler))
    with pytest.raises(ServiceError, match='ナレッジベース'):
        await kb.search(Change(summary='変更'))


@pytest.mark.parametrize('json_response', [True, False])
async def test_roundtrip_with_real_fastmcp(json_response):
    server = FastMCP('kb-mcp-test', stateless_http=True, json_response=json_response)

    @server.tool()
    async def search_documents(query: str, limit: int, mode: str,
                               scope_paths: list[str] | None, exclude_paths: list[str] | None) -> dict:
        assert query == '変更'
        return {'documents': [CANDIDATE.model_dump()], 'query_info': {}}

    @server.tool()
    async def download_document(document_id: str, encoding: str, include_images: bool) -> dict:
        assert (document_id, encoding, include_images) == ('a', 'text', False)
        return {'status': 'success', 'content': '# 仕様\n原文\n', 'encoding': encoding}

    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        kb = KnowledgeBase(Settings(), httpx.ASGITransport(app=app))
        candidates = await kb.search(Change(summary='変更'))
        assert candidates == [CANDIDATE]
        assert (await kb.download(candidates[0])).text == '# 仕様\n原文\n'
