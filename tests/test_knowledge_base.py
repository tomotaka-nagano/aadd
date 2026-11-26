import asyncio
import base64
import hashlib
import json
from io import BytesIO

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from openpyxl import Workbook

from aadd.config import Settings
from aadd.external import ServiceError
from aadd.knowledge_base import KnowledgeBase
from aadd.models import Candidate, Change


CANDIDATE = Candidate(document_id='a', file_path='仕様/spec.XLSX', title='Spec', max_score=0.8)


@pytest.fixture
def workbook_base64():
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = '機能仕様'
    sheet.append(['項目', '仕様'])
    sheet.append(['停止', '変更前'])
    with BytesIO() as stream:
        workbook.save(stream)
        return base64.b64encode(stream.getvalue()).decode('ascii')


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
        item, {**item, 'max_score': 0.9}, {**item, 'document_id': 'b', 'file_path': '仕様/old.md'},
        {**item, 'document_id': 'c', 'file_path': '仕様/x.xls'},
    ]}, structured=structured)
    settings = Settings(kb_url=url, kb_headers={'Authorization': 'Bearer test'}, scope_paths=['仕様/'],
                        exclude_paths=['仕様/旧版/'], search_mode='semantic', candidate_limit=3)
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
        'scope_paths': ['仕様/'], 'exclude_paths': ['仕様/旧版/'],
    }}
    assert all(request.headers['mcp-session-id'] == 'test-session' for request in requests[1:])
    assert requests[-1].method == 'DELETE'


async def test_download_converts_xlsx_into_markdown(workbook_base64):
    transport, requests = protocol_transport(
        {'status': 'success', 'content': workbook_base64, 'encoding': 'base64'})
    snapshot = await KnowledgeBase(Settings(), transport).download(CANDIDATE)
    assert snapshot.file_path == CANDIDATE.file_path
    assert '## 機能仕様' in snapshot.text
    assert '| 停止 | 変更前 |' in snapshot.text
    assert snapshot.sha256 == hashlib.sha256(snapshot.text.encode()).hexdigest()
    messages = [json.loads(request.content) for request in requests if request.method == 'POST']
    call = next(message for message in messages if message['method'] == 'tools/call')
    assert call['params'] == {'name': 'download_document', 'arguments': {
        'document_id': 'a', 'encoding': 'base64', 'include_images': False,
    }}


@pytest.mark.parametrize('encoding, content', [('text', '# 仕様'), ('base64', '%%%'), ('base64', 'bm90IHppcA==')])
async def test_download_rejects_non_xlsx_content(encoding, content):
    transport, _ = protocol_transport({'status': 'success', 'content': content, 'encoding': encoding})
    with pytest.raises(ServiceError):
        await KnowledgeBase(Settings(), transport).download(CANDIDATE)


@pytest.mark.parametrize('limit, message', [
    ({'max_file_bytes': 3}, 'ファイルサイズ'),
    ({'max_document_chars': 3}, '文字数上限'),
])
async def test_download_rejects_oversized_xlsx(workbook_base64, limit, message):
    transport, _ = protocol_transport({'status': 'success', 'content': workbook_base64, 'encoding': 'base64'})
    with pytest.raises(ServiceError, match=message):
        await KnowledgeBase(Settings(**limit), transport).download(CANDIDATE)


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
async def test_roundtrip_with_real_fastmcp(json_response, workbook_base64):
    server = FastMCP('kb-mcp-test', stateless_http=True, json_response=json_response)

    @server.tool()
    async def search_documents(query: str, limit: int, mode: str,
                               scope_paths: list[str] | None, exclude_paths: list[str] | None) -> dict:
        assert query == '変更'
        return {'documents': [CANDIDATE.model_dump()], 'query_info': {}}

    @server.tool()
    async def download_document(document_id: str, encoding: str, include_images: bool) -> dict:
        assert (document_id, encoding, include_images) == ('a', 'base64', False)
        return {'status': 'success', 'content': workbook_base64, 'encoding': encoding}

    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        kb = KnowledgeBase(Settings(), httpx.ASGITransport(app=app))
        candidates = await kb.search(Change(summary='変更'))
        assert candidates == [CANDIDATE]
        assert '| 停止 | 変更前 |' in (await kb.download(candidates[0])).text
