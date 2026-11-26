import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from exmgai import AsyncClient
from exmgai.decision import DecisionResponse
import httpx
import pytest
from typesafe_sdk import Noul, NoulAnswer

from aadd.ai import AI, DocumentMasks, PLAN_EXTRACT, document_masks, necessity, parse_response
from aadd.config import Settings, load_settings
from aadd.external import ServiceError
from aadd.models import Attachment, Change, Extraction, ExtractionPlan, Project, Snapshot
from aadd.models import SpecificationSearch
from aadd.redactor import MaskResult, Redactor
from aadd.usage import RunUsage, run_usage
from test_redactor import read_markdown_upload


def read_redactor_request(request):
    if not request.headers['Content-Type'].startswith('multipart/form-data'):
        return json.loads(request.content)
    manifest, file = read_markdown_upload(request)
    return {**manifest, 'upload': file, 'upload_name': manifest['parts'][0]['file']['filename'],
            'parts': [{'id': '0', 'text': file.decode('utf-8')}]}


@pytest.fixture(autouse=True)
def redaction_api(monkeypatch):
    requests = []

    def handler(request):
        body = read_redactor_request(request)
        requests.append((request.url.path, body))
        if request.url.path == '/mask':
            return httpx.Response(200, json={
                'masked_parts': [{'id': part['id'], 'masked_text': part['text']} for part in body['parts']],
                'mapping': [],
            })
        return httpx.Response(200, json={'restored_text': body['text']})

    monkeypatch.setattr('aadd.ai.Redactor', lambda settings: Redactor(settings, httpx.MockTransport(handler)))
    return requests


@pytest.fixture
def role_settings(tmp_path):
    roles = tmp_path / 'target_roles.md'
    roles.write_text('**装置状態管理**\n\n- 起動・停止・待機の状態遷移を管理します。\n', encoding='utf-8')
    return Settings(target_roles_file=roles)


def test_real_sdk_noul_contract():
    question = Noul(instructions='修正が必要ですか？')
    assert question.model_dump()['type'] == 'noul'
    response = DecisionResponse(model='microsoft-decision-1', usage={'input_tokens': 100},
                                answers={'needs_modification': NoulAnswer(noul=0.93)})
    assert necessity(response).score == 0.93
    assert necessity(response).model == 'microsoft-decision-1'


@pytest.mark.parametrize('score', [-0.1, 1.1, float('nan')])
def test_invalid_probability(score):
    with pytest.raises(ValueError):
        necessity(SimpleNamespace(answers={'needs_modification': NoulAnswer(noul=score)}, model='decision'))


def test_missing_answer():
    with pytest.raises(ValueError):
        necessity(SimpleNamespace(answers={}, model='decision'))


def test_structured_content_contract():
    result = parse_response(SimpleNamespace(content={'changes': [{'summary': '変更', 'after': '新仕様'}]}), Extraction)
    assert result.changes[0].before is None


async def test_missing_config_is_actionable(tmp_path, role_settings):
    ai = AI(role_settings.model_copy(update={'ai_config': tmp_path / 'missing.toml'}))
    with pytest.raises(ServiceError, match='AI設定がありません'):
        await ai.extract(Project(title='Test', request='変更'))


async def test_assessment_uses_sdk_question_and_probability(monkeypatch):
    response = DecisionResponse(model='microsoft-decision-1', usage={},
                                answers={'needs_modification': NoulAnswer(noul=0.91)})
    method = AsyncMock(return_value=response)
    client = SimpleNamespace(context_window=128000, decision=SimpleNamespace(system_one=method))
    ai = AI(Settings())
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='a', file_path='a.md', text='# 仕様', sha256='hash')
    assert (await ai.assess(Change(summary='変更'), document)).score == 0.91
    assert isinstance(method.call_args.args[1]['needs_modification'], Noul)


async def test_structured_invalid_response_is_retried(monkeypatch, role_settings):
    responses = [SimpleNamespace(content={'changes': [{'summary': ''}]}),
                 SimpleNamespace(content={'changes': [{'summary': '変更'}]})]
    method = AsyncMock(side_effect=responses)
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    ai = AI(role_settings)
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    result = await ai.extract(Project(title='Test', request='変更'))
    assert len(result.changes) == 1
    assert method.await_count == 2


async def test_token_budget_stops_without_truncating():
    ai = AI(Settings(max_input_tokens=1))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000)
    with pytest.raises(ServiceError, match='トークン上限'):
        await ai.budget('長い入力文です。' * 100, client)


async def test_close_releases_all_clients_only_once():
    ai = AI(Settings())
    generation, decision = AsyncMock(spec=AsyncClient), AsyncMock(spec=AsyncClient)
    ai.clients.update(generation=generation, decision=decision)
    await ai.close()
    await ai.close()
    generation.close.assert_awaited_once()
    decision.close.assert_awaited_once()
    assert not ai.clients


async def test_close_attempts_all_clients_and_surfaces_failures():
    ai = AI(Settings())
    generation, decision = AsyncMock(spec=AsyncClient), AsyncMock(spec=AsyncClient)
    failure = RuntimeError('SDK close failed')
    generation.close.side_effect = failure
    ai.clients.update(generation=generation, decision=decision)
    with pytest.raises(ExceptionGroup, match='AIクライアントの終了処理') as caught:
        await ai.close()
    assert caught.value.exceptions == (failure,)
    generation.close.assert_awaited_once()
    decision.close.assert_awaited_once()


async def test_extract_includes_roles_as_separate_scope(monkeypatch, role_settings):
    response = SimpleNamespace(content={'changes': [{'summary': '停止処理を変更'}]})
    method = AsyncMock(return_value=response)
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    ai = AI(role_settings)
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    project = Project(title='Test', request='停止処理を変更',
                      attachments=[Attachment(filename='request.md', text='詳細の変更要求')])
    await ai.extract(project)
    data = json.loads(method.call_args.args[0])
    assert data['target_roles'] == role_settings.target_roles_file.read_text(encoding='utf-8').strip()
    assert data['request'] == project.request
    assert data['attachments'][0]['text'] == '詳細の変更要求'
    assert method.call_args.kwargs['system_prompt'] == PLAN_EXTRACT
    assert '範囲外の変更は含めない' in PLAN_EXTRACT
    assert '役割記述は変更要求ではありません' in PLAN_EXTRACT


@pytest.mark.parametrize('content', [None, '', ' \n\t ', b'\xff\xfe'])
async def test_invalid_roles_stop_before_ai_call(tmp_path, monkeypatch, content):
    path = tmp_path / 'target_roles.md'
    if isinstance(content, str):
        path.write_text(content, encoding='utf-8')
    elif content is not None:
        path.write_bytes(content)
    ai = AI(Settings(target_roles_file=path))
    structured = AsyncMock()
    monkeypatch.setattr(ai, 'structured', structured)
    with pytest.raises(ServiceError, match='役割記述'):
        await ai.extract(Project(title='Test', request='変更'))
    structured.assert_not_awaited()


async def test_roles_character_limit(role_settings, monkeypatch):
    role_settings.max_input_chars = 1
    ai = AI(role_settings)
    method = AsyncMock()
    monkeypatch.setattr(ai, 'structured', method)
    with pytest.raises(ServiceError, match='役割記述が文字数上限'):
        await ai.extract(Project(title='Test', request='変更'))
    method.assert_not_awaited()


def test_role_path_is_relative_to_config(tmp_path, monkeypatch):
    config = tmp_path / 'config.toml'
    config.write_text('target_roles_file = "roles/target.md"\n', encoding='utf-8')
    monkeypatch.setenv('AADD_CONFIG', str(config))
    assert load_settings().target_roles_file == tmp_path / 'roles' / 'target.md'


async def test_no_in_scope_changes_returns_empty_without_retry(monkeypatch, role_settings):
    method = AsyncMock(return_value=SimpleNamespace(content={'changes': []}))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    ai = AI(role_settings)
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    result = await ai.extract(Project(title='Test', request='他のユニット内部の変更'))
    assert result.changes == []
    assert 'changesを空配列' in method.call_args.kwargs['system_prompt']
    method.assert_awaited_once()


def use_sensitive_redactor(ai, secret, *, fail_at=None):
    requests = []
    mapping = [{'placeholder': '[社1]', 'canonical': secret, 'category': '社名', 'surfaces': [secret],
                'confidence': 'certain', 'decided_by': 'dictionary', 'occurrences': []}]

    def handler(request):
        body = read_redactor_request(request)
        requests.append((request.url.path, body))
        if request.url.path == fail_at:
            return httpx.Response(503, json={'detail': secret})
        if request.url.path == '/mask':
            return httpx.Response(200, json={
                'masked_parts': [{'id': part['id'], 'masked_text': part['text'].replace(secret, '[社1]')}
                                 for part in body['parts']], 'mapping': mapping,
            })
        assert body['mapping'] == mapping
        return httpx.Response(200, json={'restored_text': body['text'].replace('[社1]', secret)})

    ai.redactor = Redactor(ai.settings, httpx.MockTransport(handler))
    return requests


@pytest.mark.parametrize('as_json', [False, True])
async def test_extract_masks_all_inputs_and_restores_json_values(monkeypatch, role_settings, as_json):
    secret = '秘密"社\\開発\n部'
    role_settings.target_roles_file.write_text(secret, encoding='utf-8')
    ai = AI(role_settings)
    requests = use_sensitive_redactor(ai, secret)
    content = {'changes': [{'summary': '[社1]を変更', 'before': None, 'after': '[社1]',
                            'sources': ['[社1].md'], 'unknowns': ['[社1]を確認']}]}
    method = AsyncMock(return_value=SimpleNamespace(content=json.dumps(content) if as_json else content))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    project = Project(title='Test', request=secret, attachments=[Attachment(filename=secret + '.md', text=secret)])
    result = await ai.extract(project)
    data = json.loads(method.call_args.args[0])
    assert data['target_roles'] == data['request'] == data['attachments'][0]['text'] == '[社1]'
    assert data['attachments'][0]['filename'] == '[社1].md'
    assert secret not in str(method.call_args)
    assert result.changes[0].summary == secret + 'を変更'
    assert result.changes[0].after == secret
    assert result.changes[0].before is None
    assert result.changes[0].sources == [secret + '.md']
    assert result.changes[0].unknowns == [secret + 'を確認']
    assert requests[0][0] == '/mask'
    assert all(path == '/unmask' for path, _ in requests[1:])


async def test_generate_masks_specification_and_restores_full_markdown(monkeypatch):
    secret = '秘密社'
    ai = AI(Settings())
    requests = use_sensitive_redactor(ai, secret)
    change = Change(id='c1', summary=secret + 'を変更', before=secret, after=secret + 'の新仕様')
    method = AsyncMock(return_value=SimpleNamespace(content={
        'modified': '# [社1]\n\n新仕様', 'summary': '[社1]の変更', 'change_ids': ['c1'],
    }))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='a', file_path='仕様/停止制御.v2.xlsx', text=secret, sha256='hash')
    result = await ai.generate([change], document)
    data = json.loads(method.call_args.args[0])
    assert data['specification'] == data['changes'][0]['before'] == '[社1]'
    assert secret not in str(method.call_args)
    assert result.modified == '# 秘密社\n\n新仕様'
    assert result.summary == '秘密社の変更'
    assert result.change_ids == ['c1']
    document_input = [body for path, body in requests if path == '/mask'][1]
    assert document_input['upload_name'] == '停止制御.v2.md'
    assert document_input['upload'] == secret.encode('utf-8')


async def test_decision_masks_inputs_without_unmasking(monkeypatch):
    ai = AI(Settings())
    requests = use_sensitive_redactor(ai, '秘密社')
    response = DecisionResponse(model='microsoft-decision-1', usage={},
                                answers={'needs_modification': NoulAnswer(noul=0.91)})
    method = AsyncMock(return_value=response)
    client = SimpleNamespace(context_window=128000, decision=SimpleNamespace(system_one=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='a', file_path=r'仕様\起動制御.XLSX', text='秘密社', sha256='hash')
    result = await ai.assess(Change(summary='秘密社'), document)
    data = json.loads(method.call_args.args[0])
    assert data['change']['summary'] == data['specification'] == '[社1]'
    assert '秘密社' not in str(method.call_args)
    assert result.score == 0.91
    assert [path for path, _ in requests] == ['/mask', '/mask']
    assert requests[1][1]['upload_name'] == '起動制御.md'
    assert requests[1][1]['upload'] == '秘密社'.encode('utf-8')


async def test_server_converted_document_is_masked_by_reference_and_reused():
    ai = AI(Settings(retry_count=0))
    content_hash = 'b' * 64
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body['parts'])
        return httpx.Response(200, json={
            'masked_parts': [{'id': '0', 'masked_text': '| [社1] |'}], 'mapping': [],
        })

    ai.redactor = Redactor(ai.settings, httpx.MockTransport(handler))
    doc = Snapshot(document_id='d', file_path='仕様/制御.xlsx', text='| 秘密社 |', sha256='hash',
                   redactor_content_hash=content_hash)
    token = document_masks.set(DocumentMasks())
    try:
        first = await ai.mask_document(doc)
        second = await ai.mask_document(doc)
    finally:
        document_masks.reset(token)
    assert first == second
    assert first.masked_parts[0].masked_text == '| [社1] |'
    assert calls == [[{'id': '0', 'content_hash': content_hash}]]


@pytest.mark.parametrize('operation', ['assess', 'generate'])
async def test_failed_markdown_upload_prevents_model_call(monkeypatch, operation):
    ai = AI(Settings(retry_count=0))
    calls = []

    def handler(request):
        body = read_redactor_request(request)
        calls.append(body)
        if 'upload' in body:
            return httpx.Response(503)
        return httpx.Response(200, json={
            'masked_parts': [{'id': part['id'], 'masked_text': part['text']} for part in body['parts']],
            'mapping': [],
        })

    ai.redactor = Redactor(ai.settings, httpx.MockTransport(handler))
    method = AsyncMock()
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000,
                             chat=SimpleNamespace(create=method), decision=SimpleNamespace(system_one=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='a', file_path='a.md', text='# 仕様', sha256='hash')
    with pytest.raises(ServiceError, match='マスクAPI'):
        change = Change(summary='変更')
        if operation == 'generate':
            await ai.generate([change], document)
        else:
            await ai.assess(change, document)
    assert len(calls) == 2 and calls[1]['upload'] == document.text.encode('utf-8')
    method.assert_not_awaited()


async def test_standalone_document_mask_is_reused_and_conflicting_labels_restore(monkeypatch):
    ai = AI(Settings())
    batches = []

    def handler(request):
        body = read_redactor_request(request)
        if request.url.path == '/mask':
            batches.append((body['parts'], body.get('upload_name')))
            text = '文書の社名' if body['parts'] == [{'id': '0', 'text': '文書の社名'}] else '要求の社名'
            return httpx.Response(200, json={
                'masked_parts': [{'id': p['id'], 'masked_text': p['text'].replace(text, '[社1]')}
                                 for p in body['parts']],
                'mapping': [{'placeholder': '[社1]', 'canonical': text}],
            })
        text = body['text']
        for item in body['mapping']:
            text = text.replace(item['placeholder'], item['canonical'])
        return httpx.Response(200, json={'restored_text': text})

    async def generate(message, **kwargs):
        data = json.loads(message)
        assert data['specification'] != data['changes'][0]['summary']
        assert '文書の社名' not in message and '要求の社名' not in message
        return SimpleNamespace(content={'modified': data['specification'], 'summary': data['changes'][0]['summary'],
                                        'change_ids': ['c']})

    ai.redactor = Redactor(ai.settings, httpx.MockTransport(handler))
    response = DecisionResponse(model='decision', usage={'input_tokens': 100},
                                answers={'needs_modification': NoulAnswer(noul=0.9)})
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=generate),
                             decision=SimpleNamespace(system_one=AsyncMock(return_value=response)))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    doc = Snapshot(document_id='d', file_path='d.md', sha256='hash', text='文書の社名')
    change = Change(id='c', summary='要求の社名')
    token = document_masks.set(DocumentMasks())
    try:
        await ai.assess(change, doc)
        await ai.assess(change, doc)
        result = await ai.generate([change], doc)
        await ai.assess(change, doc.model_copy(update={'file_path': '仕様/別仕様.xlsx'}))
    finally:
        document_masks.reset(token)
    assert [(batch, name) for batch, name in batches if name is not None] == [
        ([{'id': '0', 'text': '文書の社名'}], 'd.md'),
        ([{'id': '0', 'text': '文書の社名'}], '別仕様.md'),
    ]
    assert all(not any(part['text'] == '文書の社名' for part in batch)
               for batch, _ in batches if len(batch) > 1)
    assert (result.modified, result.summary) == ('文書の社名', '要求の社名')


async def test_usage_counts_model_response_before_failed_validation(monkeypatch, role_settings):
    role_settings.retry_count = 0
    ai = AI(role_settings)
    ai.rates[role_settings.generation_model] = (2.0, 4.0)
    method = AsyncMock(return_value=SimpleNamespace(content={'changes': [{'summary': ''}]},
                                                    input_tokens=100, output_tokens=50))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    usage = RunUsage()
    token = run_usage.set(usage)
    try:
        with pytest.raises(ServiceError):
            await ai.extract(Project(title='Test', request='変更'))
    finally:
        run_usage.reset(token)
    assert usage.generation.calls == 1
    assert usage.generation.estimated_usd == pytest.approx(0.0004)
    assert usage.generation.unpriced_calls == 0


async def test_refine_sends_masked_excerpt_and_resolves_valid_evidence(monkeypatch, role_settings):
    ai = AI(role_settings)
    requests = use_sensitive_redactor(ai, '秘密社')
    document = Snapshot(document_id='d', file_path='秘密社.md', sha256='hash',
                        text='# 仕様\n\n' + ('無関係な行\n' * 50) + '停止時間は秘密社構成で1秒')
    draft = ExtractionPlan(changes=[Change(summary='時間変更', after='2秒')],
                           searches=[SpecificationSearch(query='停止時間', question='現状は何秒か')])

    async def reply(message, **kwargs):
        data = json.loads(message)
        assert '秘密社' not in message and '秘密社.md' not in message
        assert len(data['evidence']) == 1
        assert len(data['evidence'][0]['excerpt'].splitlines()) < 10
        assert '停止時間は[社1]構成で1秒' in data['evidence'][0]['excerpt']
        return SimpleNamespace(content={'changes': [{
            'summary': '時間変更', 'before': '1秒', 'after': '2秒', 'sources': ['変更要求'],
            'evidence_ids': ['kb_1'],
        }]})

    method = AsyncMock(side_effect=reply)
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    result = await ai.refine({'request': '停止時間を2秒に変更'}, draft, [('kb_1', document, '停止時間')])
    assert result.changes[0].before == '1秒' and result.changes[0].after == '2秒'
    assert result.changes[0].sources[0] == '変更要求'
    assert result.changes[0].sources[1] == '秘密社.md（現行仕様）'
    document_input = [body for path, body in requests if path == '/mask'][1]
    assert document_input['upload_name'] == '秘密社.md'
    assert document_input['upload'] == document.text.encode('utf-8')


async def test_refine_rejects_unsent_document_reference(monkeypatch, role_settings):
    role_settings.retry_count = 0
    ai = AI(role_settings)
    response = SimpleNamespace(content={'changes': [{'summary': '時間変更', 'evidence_ids': ['not_sent']}]})
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000,
                             chat=SimpleNamespace(create=AsyncMock(return_value=response)))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='d', file_path='d.md', sha256='hash', text='停止時間は1秒')
    with pytest.raises(ServiceError, match='生成AIの処理'):
        await ai.refine({'request': '時間変更'}, ExtractionPlan(changes=[]), [('kb_1', document, '停止時間')])


@pytest.mark.parametrize('operation', ['extract', 'generate', 'assess'])
async def test_mask_failure_prevents_model_call(monkeypatch, role_settings, operation):
    role_settings.retry_count = 0
    ai = AI(role_settings)
    use_sensitive_redactor(ai, '秘密社', fail_at='/mask')
    method = AsyncMock()
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000,
                             chat=SimpleNamespace(create=method), decision=SimpleNamespace(system_one=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    document = Snapshot(document_id='a', file_path='a.md', text='秘密社', sha256='hash')
    with pytest.raises(ServiceError, match='マスクAPI') as error:
        if operation == 'extract':
            await ai.extract(Project(title='Test', request='秘密社'))
        elif operation == 'generate':
            await ai.generate([Change(summary='秘密社')], document)
        else:
            await ai.assess(Change(summary='秘密社'), document)
    assert '秘密社' not in str(error.value)
    method.assert_not_awaited()


async def test_unmask_failure_does_not_return_masked_result(monkeypatch, role_settings):
    role_settings.retry_count = 0
    ai = AI(role_settings)
    use_sensitive_redactor(ai, '秘密社', fail_at='/unmask')
    method = AsyncMock(return_value=SimpleNamespace(content={'changes': [{'summary': '[社1]'}]}))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    with pytest.raises(ServiceError, match='マスクAPI'):
        await ai.extract(Project(title='Test', request='秘密社'))
    method.assert_awaited_once()


@pytest.mark.parametrize('operation', ['generate', 'assess'])
async def test_masked_input_budget_is_checked_before_model_call(monkeypatch, operation):
    ai = AI(Settings(max_input_tokens=2000))
    method = AsyncMock()
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000,
                             chat=SimpleNamespace(create=method), decision=SimpleNamespace(system_one=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))

    def mask(texts):
        parts = [{'id': str(index), 'masked_text': '機密' * 10000} for index in range(len(texts))]
        return MaskResult(masked_parts=parts, mapping=[])

    monkeypatch.setattr(ai.redactor, 'mask', AsyncMock(side_effect=mask))
    monkeypatch.setattr(ai.redactor, 'mask_markdown', AsyncMock(side_effect=lambda text, *, filename: mask([text])))
    document = Snapshot(document_id='a', file_path='a.md', text='仕様', sha256='hash')
    with pytest.raises(ServiceError, match='トークン上限'):
        if operation == 'generate':
            await ai.generate([Change(summary='変更')], document)
        else:
            await ai.assess(Change(summary='変更'), document)
    ai.redactor.mask.assert_awaited_once()
    ai.redactor.mask_markdown.assert_awaited_once_with(document.text, filename='a.md')
    method.assert_not_awaited()


async def test_preview_uses_extraction_input_without_ai_config_or_model_call(monkeypatch, role_settings, tmp_path):
    role_settings.ai_config = tmp_path / 'missing.toml'
    ai = AI(role_settings)
    requests = use_sensitive_redactor(ai, '秘密社')
    build = AsyncMock(side_effect=AssertionError('プレビューではモデルを生成しない'))
    monkeypatch.setattr(ai, 'client', build)
    project = Project(title='Test', request='秘密社の変更',
                      attachments=[Attachment(filename='秘密社.md', text='秘密社の仕様')])
    preview = await ai.preview_extraction(project)
    build.assert_not_awaited()
    assert [path for path, _ in requests] == ['/mask']
    assert '秘密社' not in preview.model_dump_json()
    assert 'mapping' not in preview.model_dump()
    assert json.loads(preview.message)['attachments'][0]['text'] == '[社1]の仕様'
    assert preview.response_schema == ExtractionPlan.model_json_schema()
    method = AsyncMock(return_value=SimpleNamespace(content={'changes': []}))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    await ai.extract(project)
    assert method.call_args.args[0] == preview.message
    assert method.call_args.kwargs['system_prompt'] == preview.system_prompt


async def test_preview_rejects_missing_roles_before_masking(monkeypatch, tmp_path, redaction_api):
    ai = AI(Settings(target_roles_file=tmp_path / 'missing.md'))
    build = AsyncMock()
    monkeypatch.setattr(ai, 'client', build)
    with pytest.raises(ServiceError, match='役割記述'):
        await ai.preview_extraction(Project(title='Test', request='変更'))
    build.assert_not_awaited()
    assert redaction_api == []
