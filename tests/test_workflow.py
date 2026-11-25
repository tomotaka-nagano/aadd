import asyncio
import hashlib
import json

import pytest

from aadd.config import Settings, load_settings
from aadd.external import ServiceError
from aadd.models import Assessment, Candidate, Change, ChangeUpdate, Extraction, ExtractionPlan, Generated, Project
from aadd.models import Refinement, Snapshot, SpecificationSearch
from aadd.storage import Conflict, Store
from aadd.workflow import Workflow, choose, markdown_diff, update_changes


class FakeAI:
    def __init__(self):
        self.generated = []
        self.assessed = []
        self.fail_once = False
        self.closed = False

    async def close(self):
        self.closed = True

    async def extract(self, project):
        return Extraction(changes=[Change(summary='時間変更', before='1秒', after='2秒')])

    async def extraction_data(self, project):
        return {'project': project}

    async def plan_extraction(self, data):
        return ExtractionPlan(changes=(await self.extract(data['project'])).changes)

    async def assess(self, change, document):
        self.assessed.append((change.id, document.document_id))
        if self.fail_once and document.document_id == 'b':
            self.fail_once = False
            raise RuntimeError('unavailable')
        return Assessment(score=0.9 if document.document_id == 'a' else 0.6, model='decision')

    async def generate(self, changes, document):
        self.generated.append([change.id for change in changes])
        return Generated(modified='# 仕様\n2秒\n', summary='時間変更', change_ids=[change.id for change in changes])


class FakeKB:
    def __init__(self):
        self.downloaded = []

    async def search(self, change):
        return [Candidate(document_id=key, file_path=f'{key}.md', title=key, max_score=0.8) for key in ['a', 'b']]

    async def download(self, candidate):
        self.downloaded.append(candidate.document_id)
        text = '# 仕様\n1秒\n'
        return Snapshot(document_id=candidate.document_id, file_path=candidate.file_path,
                        text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


async def finish(engine):
    await asyncio.gather(*list(engine.tasks))


def searches(*queries):
    return [SpecificationSearch(query=query, question=f'{query}の既存仕様を確認') for query in queries]


class RefiningAI(FakeAI):
    def __init__(self):
        super().__init__()
        self.rounds = []

    async def plan_extraction(self, data):
        return ExtractionPlan(changes=[Change(summary='時間変更', after='2秒')], searches=searches('q1', 'q2'))

    async def search_query(self, item):
        return item.query

    async def mask_document(self, document):
        pass

    async def refine(self, data, draft, evidence):
        self.rounds.append([key for key, _, _ in evidence])
        step = len(self.rounds)
        return Refinement.model_validate({
            'changes': [{'summary': '時間変更', 'after': '2秒', 'before': f'{step}秒', 'evidence_ids': [evidence[-1][0]]}],
            'searches': searches('q1', f'q{step * 2 + 1}', f'q{step * 2 + 2}'),
        })


class RefiningKB(FakeKB):
    def __init__(self):
        super().__init__()
        self.queries = []

    async def search_query(self, query):
        self.queries.append(query)
        return [Candidate(document_id=key, file_path=f'{key}.md', title=key, max_score=0.8)
                for key in [query, 'shared']]


async def test_requirement_refinement_has_bounded_rounds_and_does_not_seed_generation(tmp_path, capsys):
    ai, kb = RefiningAI(), RefiningKB()
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更'))
    engine = Workflow(store, Settings(), ai, kb)
    run = await engine.start(project.id, 'extract', 0)
    await finish(engine)
    saved, result = store.find_run(run.id)
    assert saved.changes[0].before == '3秒'
    assert saved.changes[0].after == '2秒'
    assert kb.queries == ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
    assert kb.downloaded == kb.queries
    assert list(map(len, ai.rounds)) == [2, 4, 6]
    assert result.extraction_warnings and not result.documents and not result.candidates
    assert not ai.assessed
    generation = await engine.start(project.id, 'generate', 1)
    await finish(engine)
    assert kb.downloaded[-2:] == ['a', 'b']
    assert store.find_run(generation.id)[1].status == 'completed'
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line['run_id'] for line in logs] == [run.id, generation.id]


@pytest.mark.parametrize(('rounds', 'per_round', 'total', 'expected'), [
    (4, 1, 8, [1, 2, 3, 4]),
    (4, 3, 5, [3, 5]),
])
async def test_requirement_refinement_obeys_toml_limits(tmp_path, monkeypatch, rounds, per_round, total, expected):
    config = tmp_path / 'config.toml'
    config.write_text(
        f'refinement_max_rounds = {rounds}\n'
        f'refinement_documents_per_round = {per_round}\n'
        f'refinement_max_documents = {total}\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('AADD_CONFIG', str(config))
    ai, kb = RefiningAI(), RefiningKB()
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更'))
    engine = Workflow(store, load_settings(), ai, kb)
    run = await engine.start(project.id, 'extract', 0)
    await finish(engine)
    assert store.find_run(run.id)[1].status == 'completed'
    assert list(map(len, ai.rounds)) == expected
    assert len(kb.downloaded) == expected[-1]


async def test_requirement_refinement_without_kb_keeps_draft_with_warning(tmp_path):
    class UnavailableKB(RefiningKB):
        async def search_query(self, query):
            raise ServiceError('KB offline')

    ai = RefiningAI()
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更'))
    engine = Workflow(store, Settings(), ai, UnavailableKB())
    run = await engine.start(project.id, 'extract', 0)
    await finish(engine)
    saved, result = store.find_run(run.id)
    assert saved.changes[0].after == '2秒' and saved.changes[0].before is None
    assert result.status == 'completed' and any('失敗' in warning for warning in result.extraction_warnings)
    assert ai.rounds == []


async def test_empty_scope_extraction_completes_but_cannot_generate(tmp_path):
    class EmptyAI(FakeAI):
        async def extract(self, project):
            return Extraction(changes=[])

    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='他のユニット内部の変更'))
    engine = Workflow(store, Settings(), EmptyAI(), FakeKB())
    run = await engine.start(project.id, 'extract', 0)
    await finish(engine)
    updated = store.get(project.id)
    assert updated.changes == [] and updated.revision == 1
    assert store.find_run(run.id)[1].status == 'completed'
    with pytest.raises(ValueError, match='変更内容を抽出・確認'):
        await engine.start(project.id, 'generate', 1)
    with pytest.raises(ValueError):
        ChangeUpdate(revision=1, changes=[])


async def test_extract_and_generate_grouped(tmp_path):
    store = Store(tmp_path)
    ai, kb = FakeAI(), FakeKB()
    engine = Workflow(store, Settings(), ai, kb)
    project = store.create(Project(title='Test', request='変更'))
    await engine.start(project.id, 'extract', 0)
    await finish(engine)
    project = store.get(project.id)
    assert project.revision == 1
    assert project.changes[0].after == '2秒'
    project.changes.append(Change(summary='関連変更'))
    store.update(project.id, lambda current: setattr(current, 'changes', project.changes))
    run = await engine.start(project.id, 'generate', 1)
    await finish(engine)
    _, completed = store.find_run(run.id)
    assert completed.status == 'completed'
    assert len(completed.proposals) == 1
    assert len(ai.generated[0]) == 2
    assert kb.downloaded == ['a', 'b']
    proposal = completed.proposals['a']
    assert '-1秒' in proposal.diff and '+2秒' in proposal.diff
    assert proposal.original == completed.documents['a'].text


async def test_partial_assessment_stops_generation_then_resumes(tmp_path):
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更', changes=[Change(summary='変更')]))
    ai = FakeAI()
    ai.fail_once = True
    engine = Workflow(store, Settings(), ai, FakeKB())
    run = await engine.start(project.id, 'generate', 0)
    await finish(engine)
    _, partial = store.find_run(run.id)
    assert partial.status == 'review'
    assert not partial.proposals and not ai.generated
    await engine.retry(run.id)
    await finish(engine)
    assert store.find_run(run.id)[1].status == 'completed'
    assert ai.assessed.count((project.changes[0].id, 'a')) == 1


async def test_no_candidates_requires_review(tmp_path):
    class EmptyKB(FakeKB):
        async def search(self, change):
            return []

    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更', changes=[Change(summary='変更')]))
    engine = Workflow(store, Settings(), FakeAI(), EmptyKB())
    run = await engine.start(project.id, 'generate', 0)
    await finish(engine)
    assert store.find_run(run.id)[1].status == 'review'
    assert not store.find_run(run.id)[1].proposals


def test_tie_break_and_unnecessary():
    candidates = [Candidate(document_id=key, file_path=f'{key}.md', title=key, max_score=0.8) for key in ['b', 'a']]
    assessments = {key: Assessment(score=0.8, model='decision') for key in ['b', 'a']}
    assert choose(candidates, assessments, 0.5) == 'a'
    assert choose(candidates, assessments, 0.8) is None


def test_revision_and_change_ids():
    project = Project(title='Test', request='変更')
    update_changes(project, ChangeUpdate(revision=0, changes=[Change(summary='新変更')]))
    assert project.revision == 1
    with pytest.raises(Conflict):
        update_changes(project, ChangeUpdate(revision=0, changes=[Change(summary='古い変更')]))
    with pytest.raises(ValueError):
        ChangeUpdate(revision=1, changes=[Change(id='same', summary='a'), Change(id='same', summary='b')])


def test_diff_handles_missing_newline():
    diff = markdown_diff('old', 'new', 'a.md')
    assert '-old\n\\ No newline at end of file\n+new\n' in diff


async def test_restart_and_stale_retry(tmp_path):
    class BrokenAI(FakeAI):
        async def extract(self, project):
            raise RuntimeError('secret token')

    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更'))
    engine = Workflow(store, Settings(), BrokenAI(), FakeKB())
    run = await engine.start(project.id, 'extract', 0)
    await finish(engine)
    assert 'secret token' not in str(store.find_run(run.id)[1].issues)
    restarted = Workflow(Store(tmp_path), Settings(), FakeAI(), FakeKB())
    await restarted.retry(run.id)
    await finish(restarted)
    assert restarted.store.get(project.id).changes
    with pytest.raises(Conflict):
        await restarted.retry(run.id)


async def test_retry_regenerates_proposal_when_new_change_joins_document(tmp_path):
    class PartialAI(FakeAI):
        block_second = True

        async def assess(self, change, document):
            if change.id == 'second' and self.block_second:
                raise RuntimeError('unavailable')
            return await super().assess(change, document)

    store = Store(tmp_path)
    changes = [Change(id='first', summary='変更1'), Change(id='second', summary='変更2')]
    project = store.create(Project(title='Test', request='変更', changes=changes))
    ai = PartialAI()
    engine = Workflow(store, Settings(), ai, FakeKB())
    run = await engine.start(project.id, 'generate', 0)
    await finish(engine)
    assert store.find_run(run.id)[1].proposals['a'].change_ids == ['first']
    ai.block_second = False
    await engine.retry(run.id)
    await finish(engine)
    assert store.find_run(run.id)[1].proposals['a'].change_ids == ['first', 'second']
    assert len(ai.generated) == 2


@pytest.mark.parametrize('mode', ['unchanged', 'missing_id', 'duplicate_id'])
async def test_invalid_proposal_requires_review(tmp_path, mode):
    class InvalidAI(FakeAI):
        async def generate(self, changes, document):
            ids = [change.id for change in changes]
            modified = document.text if mode == 'unchanged' else '# changed'
            ids = ['wrong'] if mode == 'missing_id' else ids * 2 if mode == 'duplicate_id' else ids
            return Generated(modified=modified, summary='変更', change_ids=ids)

    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更', changes=[Change(summary='変更')]))
    engine = Workflow(store, Settings(), InvalidAI(), FakeKB())
    run = await engine.start(project.id, 'generate', 0)
    await finish(engine)
    completed = store.find_run(run.id)[1]
    assert completed.status == 'review'
    assert not completed.proposals


async def test_cancel_and_restart_preserves_checkpoint(tmp_path):
    class WaitingAI(FakeAI):
        async def assess(self, change, document):
            await asyncio.Event().wait()

    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='変更', changes=[Change(summary='変更')]))
    engine = Workflow(store, Settings(), WaitingAI(), FakeKB())
    run = await engine.start(project.id, 'generate', 0)
    while not store.find_run(run.id)[1].documents:
        await asyncio.sleep(0)
    with pytest.raises(Conflict):
        await engine.start(project.id, 'generate', 0)
    await engine.close()
    assert engine.ai.closed
    assert store.find_run(run.id)[1].status == 'interrupted'
    restarted = Workflow(Store(tmp_path), Settings(), FakeAI(), FakeKB())
    await restarted.retry(run.id)
    await finish(restarted)
    assert restarted.store.find_run(run.id)[1].status == 'completed'


async def test_close_waits_for_tasks_before_releasing_ai(tmp_path):
    class ClosingAI(FakeAI):
        async def close(self):
            assert stopped.is_set()
            await super().close()

    started, stopped = asyncio.Event(), asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            stopped.set()

    ai = ClosingAI()
    engine = Workflow(Store(tmp_path), Settings(), ai, FakeKB())
    task = asyncio.create_task(operation())
    engine.tasks.add(task)
    await started.wait()
    await engine.close()
    assert task.cancelled()
    assert ai.closed
