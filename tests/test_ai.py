import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from exmgai.decision import DecisionResponse
import pytest
from typesafe_sdk import Noul, NoulAnswer

from aadd.ai import AI, EXTRACT, necessity, parse_response
from aadd.config import Settings, load_settings
from aadd.external import ServiceError
from aadd.models import Attachment, Change, Extraction, Project, Snapshot


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
    assert method.call_args.kwargs['system_prompt'] == EXTRACT
    assert '範囲外の変更は含めない' in EXTRACT
    assert '役割記述は変更要求ではありません' in EXTRACT


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
