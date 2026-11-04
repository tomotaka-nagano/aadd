from types import SimpleNamespace
from unittest.mock import AsyncMock

from exmgai.decision import DecisionResponse
import pytest
from typesafe_sdk import Noul, NoulAnswer

from aadd.ai import AI, necessity, parse_response
from aadd.config import Settings
from aadd.external import ServiceError
from aadd.models import Change, Extraction, Project, Snapshot


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


async def test_missing_config_is_actionable(tmp_path):
    ai = AI(Settings(ai_config=tmp_path / 'missing.toml'))
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


async def test_structured_invalid_response_is_retried(monkeypatch):
    responses = [SimpleNamespace(content={'changes': []}),
                 SimpleNamespace(content={'changes': [{'summary': '変更'}]})]
    method = AsyncMock(side_effect=responses)
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000, chat=SimpleNamespace(create=method))
    ai = AI(Settings(retry_count=1))
    monkeypatch.setattr(ai, 'client', AsyncMock(return_value=client))
    result = await ai.extract(Project(title='Test', request='変更'))
    assert len(result.changes) == 1
    assert method.await_count == 2


async def test_token_budget_stops_without_truncating():
    ai = AI(Settings(max_input_tokens=1))
    client = SimpleNamespace(context_window=128000, max_output_tokens=16000)
    with pytest.raises(ServiceError, match='トークン上限'):
        await ai.budget('長い入力文です。' * 100, client)
