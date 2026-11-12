import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

from aadd.ai import AI, EXTRACT
from aadd.config import Settings
from aadd.main import create_app
from aadd.models import Change, Extraction, MaskPreview, Project, Run
from aadd.redactor import Redactor
from test_workflow import FakeAI, FakeKB


def browser_app():
    directory = TemporaryDirectory(prefix='aadd-browser-')
    application = create_app(Settings(data_dir=Path(directory.name)), ai=BrowserAI(), kb=FakeKB())
    application.state.test_directory = directory
    return application


class BrowserAI(FakeAI):
    async def preview_extraction(self, project):
        data = {'target_roles': '装置の状態管理', 'request': project.request,
                'attachments': [item.model_dump() for item in project.attachments]}
        message = json.dumps(data, ensure_ascii=False).replace('秘密社', '[社1]')
        return MaskPreview(message=message, system_prompt=EXTRACT, response_schema=Extraction.model_json_schema())


@pytest.fixture
def client(tmp_path):
    application = create_app(Settings(data_dir=tmp_path, ai_config=tmp_path / 'missing.toml'), ai=FakeAI(), kb=FakeKB())
    with TestClient(application) as test_client:
        yield test_client


def test_health_without_settings(client):
    assert client.get('/api/health').json() == {'status': 'ok', 'ai_configured': False}


def test_static_application(client):
    response = client.get('/')
    assert response.status_code == 200
    assert '変更要求' in response.text
    assert '<title>AI aided design environment.</title>' in response.text
    assert '<span>AI aided design environment.</span>' in response.text
    assert 'brand-caption' not in response.text
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/style.css').status_code == 200


def test_create_edit_and_delete(client):
    response = client.post('/api/projects', data={'request': '変更要求'}, files={'files': ('input.md', b'# spec')})
    assert response.status_code == 201
    project = response.json()
    assert project['attachments'][0]['text'] == '# spec'
    changes = {'revision': 0, 'changes': [{'id': 'change', 'summary': '変更', 'after': '新仕様'}]}
    updated = client.put(f'/api/projects/{project["id"]}/changes', json=changes)
    assert updated.json()['revision'] == 1
    assert client.put(f'/api/projects/{project["id"]}/changes', json=changes).status_code == 409
    assert client.get('/api/projects').json()[0]['id'] == project['id']
    assert client.delete(f'/api/projects/{project["id"]}').status_code == 204
    assert client.get(f'/api/projects/{project["id"]}').status_code == 404


def test_invalid_input_and_origin(client):
    assert client.post('/api/projects').status_code == 422
    response = client.post('/api/projects', data={'request': '変更'}, headers={'Origin': 'http://attacker'})
    assert response.status_code == 403
    assert client.post('/api/projects', files={'files': ('old.doc', b'old')}).status_code == 422
    assert client.get('/api/projects', headers={'Host': 'attacker'}).status_code == 400


def test_busy_project_rejects_mutation(client):
    store = client.app.state.store
    project = store.create(Project(title='Test', request='変更', runs=[Run(kind='extract', revision=0)]))
    assert client.delete(f'/api/projects/{project.id}').status_code == 409
    response = client.post(f'/api/projects/{project.id}/runs', json={'revision': 0})
    assert response.status_code == 409


def test_run_completion_and_download(client):
    store = client.app.state.store
    project = store.create(Project(title='Test', request='変更', changes=[Change(summary='時間変更')]))
    response = client.post(f'/api/projects/{project.id}/runs', json={'revision': 0})
    assert response.status_code == 202
    run_id = response.json()['id']
    client.portal.call(wait_for_tasks, client.app.state.workflow)
    run = client.get(f'/api/runs/{run_id}').json()['run']
    assert run['status'] == 'completed'
    proposal = next(iter(run['proposals'].values()))
    url = f'/api/projects/{project.id}/proposals/{proposal["id"]}'
    assert client.get(url).json()['modified'] == '# 仕様\n2秒\n'
    assert client.get(url + '/download?format=diff').text == proposal['diff']
    assert client.get(url + '/download?format=bad').status_code == 422


async def wait_for_tasks(engine):
    import asyncio
    await asyncio.gather(*list(engine.tasks))


def test_combined_input_limit(client):
    client.app.state.settings.max_input_chars = 5
    response = client.post('/api/projects', data={'request': 'ab'}, files={'files': ('a.txt', b'cdef')})
    assert response.status_code == 413


def test_streaming_request_limit(tmp_path):
    settings = Settings(data_dir=tmp_path, max_file_bytes=1, max_files=1, max_input_chars=1)
    with TestClient(create_app(settings)) as limited:
        response = limited.post('/api/projects', files={'files': ('a.txt', b'x' * (1024 * 1024 + 100))})
        assert response.status_code == 413


@pytest.fixture
def preview_client(client, tmp_path):
    roles = tmp_path / 'roles.md'
    roles.write_text('秘密社の装置状態管理', encoding='utf-8')
    settings = client.app.state.settings
    settings.target_roles_file = roles
    settings.retry_count = 0
    ai = AI(settings)
    ai.client = AsyncMock(side_effect=AssertionError('プレビューでモデルを呼んではいけない'))
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert request.url.path == '/mask'
        return httpx.Response(200, json={
            'masked_parts': [{'id': part['id'], 'masked_text': part['text'].replace('秘密社', '[社1]')}
                             for part in body['parts']],
            'mapping': [{'placeholder': '[社1]', 'canonical': '秘密社'}],
        })

    ai.redactor = Redactor(settings, httpx.MockTransport(handler))
    client.app.state.workflow.ai = ai
    return client, ai, requests


def test_preview_unsaved_request_and_attachment_without_persisting(preview_client):
    client, ai, requests = preview_client
    response = client.post('/api/extraction-previews', data={'request': '秘密社の仕様を変更'},
                           files={'files': ('秘密社.md', '秘密社の資料'.encode())})
    assert response.status_code == 200
    assert '秘密社' not in response.text
    assert 'mapping' not in response.json()
    data = json.loads(response.json()['message'])
    assert data['request'] == '[社1]の仕様を変更'
    assert data['target_roles'] == '[社1]の装置状態管理'
    assert data['attachments'][0]['filename'] == '[社1].md'
    assert data['attachments'][0]['text'] == '[社1]の資料'
    assert client.get('/api/projects').json() == []
    assert not (client.app.state.settings.data_dir / 'attachments').exists()
    assert [path for path, _ in requests] == ['/mask']
    ai.client.assert_not_awaited()


def test_preview_saved_project_leaves_changes_and_runs_untouched(preview_client):
    client, ai, _ = preview_client
    project = client.post('/api/projects', data={'request': '秘密社の仕様を変更'}).json()
    before = client.get(f'/api/projects/{project["id"]}').json()
    response = client.post(f'/api/projects/{project["id"]}/extraction-previews', json={'revision': 0})
    assert response.status_code == 200
    assert json.loads(response.json()['message'])['request'] == '[社1]の仕様を変更'
    assert client.get(f'/api/projects/{project["id"]}').json() == before
    ai.client.assert_not_awaited()


def test_preview_rejects_invalid_input_limits_and_unsupported_attachments(preview_client):
    client, ai, requests = preview_client
    assert client.post('/api/extraction-previews').status_code == 422
    assert client.post('/api/extraction-previews', files={'files': ('old.doc', b'old')}).status_code == 422
    client.app.state.settings.max_input_chars = 5
    assert client.post('/api/extraction-previews', data={'request': 'abcdef'}).status_code == 413
    assert client.post('/api/extraction-previews', data={'request': 'ab'},
                       files={'files': ('a.txt', b'cdef')}).status_code == 413
    assert requests == []
    ai.client.assert_not_awaited()


def test_preview_saved_project_rejects_missing_busy_and_stale(preview_client):
    client, ai, requests = preview_client
    store = client.app.state.store
    project = store.create(Project(title='Test', request='変更', revision=1))
    url = f'/api/projects/{project.id}/extraction-previews'
    assert client.post(url, json={'revision': 0}).status_code == 409
    assert client.post('/api/projects/missing/extraction-previews', json={'revision': 0}).status_code == 404
    busy = store.create(Project(title='Test', request='変更', runs=[Run(kind='extract', revision=0)]))
    assert client.post(f'/api/projects/{busy.id}/extraction-previews', json={'revision': 0}).status_code == 409
    assert requests == []
    ai.client.assert_not_awaited()


def test_preview_api_error_is_explicit_and_does_not_fall_back(preview_client):
    client, ai, _ = preview_client
    ai.redactor = Redactor(ai.settings, httpx.MockTransport(lambda request: httpx.Response(503)))
    response = client.post('/api/extraction-previews', data={'request': '秘密社の仕様を変更'})
    assert response.status_code == 422
    assert 'マスクAPI' in response.json()['detail']
    assert '秘密社' not in response.text
    assert client.get('/api/projects').json() == []
    ai.client.assert_not_awaited()
