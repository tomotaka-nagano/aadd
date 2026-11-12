import asyncio
import hashlib

import pytest

from aadd.config import Settings
from aadd.models import Assessment, Candidate, Change, ChangeUpdate, Extraction, Generated, Project, Snapshot
from aadd.storage import Conflict, Store
from aadd.workflow import Workflow, choose, markdown_diff, update_changes


class FakeAI:
    def __init__(self):
        self.generated = []
        self.assessed = []
        self.fail_once = False

    async def extract(self, project):
        return Extraction(changes=[Change(summary='時間変更', before='1秒', after='2秒')])

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
    assert store.find_run(run.id)[1].status == 'interrupted'
    restarted = Workflow(Store(tmp_path), Settings(), FakeAI(), FakeKB())
    await restarted.retry(run.id)
    await finish(restarted)
    assert restarted.store.find_run(run.id)[1].status == 'completed'
