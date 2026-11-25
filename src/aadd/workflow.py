import asyncio
from collections import defaultdict
from difflib import unified_diff
import unicodedata

from aadd.ai import AI, DocumentMasks, document_masks
from aadd.config import Settings
from aadd.external import ServiceError
from aadd.knowledge_base import KnowledgeBase
from aadd.models import Change, ChangeUpdate, Project, Proposal, Run, identifier
from aadd.storage import Conflict, Store, editable, finish_write
from aadd.usage import RunUsage, run_usage


MAX_QUERIES_PER_ROUND = 2


def get_run(project: Project, run_id: str) -> Run:
    return next(run for run in project.runs if run.id == run_id)


def markdown_diff(original: str, modified: str, path: str) -> str:
    lines = unified_diff(original.splitlines(keepends=True), modified.splitlines(keepends=True),
                         fromfile=f'a/{path}', tofile=f'b/{path}')
    return ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)


def update_changes(project: Project, update: ChangeUpdate) -> None:
    editable(project)
    if project.revision != update.revision:
        raise Conflict('別の画面で変更されています。案件を再読込みしてください。')
    project.changes = update.changes
    project.revision += 1


def begin_run(project: Project, kind: str, revision: int) -> Run:
    editable(project)
    if revision != project.revision:
        raise Conflict('古い版です。案件を再読込みしてください。')
    if kind == 'generate' and not project.changes:
        raise ValueError('変更内容を抽出・確認してください。')
    run = Run(kind=kind, revision=revision, changes=[item.model_copy(deep=True) for item in project.changes])
    project.runs.append(run)
    return run


def choose(candidates, assessments, threshold: float) -> str | None:
    ordered = sorted(candidates, key=lambda item: (
        -assessments[item.document_id].score, -item.max_score, item.document_id,
    ))
    if ordered and assessments[ordered[0].document_id].score > threshold:
        return ordered[0].document_id
    return None


class Workflow:
    def __init__(self, store: Store, settings: Settings, ai=None, kb=None):
        self.store = store
        self.settings = settings
        self.ai = ai or AI(settings)
        self.kb = kb or KnowledgeBase(settings)
        self.tasks: set[asyncio.Task] = set()
        self.write_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(settings.concurrency)

    async def start(self, project_id: str, kind: str, revision: int) -> Run:
        project = await asyncio.to_thread(self.store.update, project_id,
                                          lambda item: begin_run(item, kind, revision))
        run = project.runs[-1]
        self.launch(project, run)
        return run

    def launch(self, project: Project, run: Run) -> None:
        task = asyncio.create_task(self.execute(project, run))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def checkpoint(self, project: Project, run: Run) -> None:
        async with self.write_lock:
            snapshot = run.model_copy(deep=True)

            def save(current):
                index = next(index for index, item in enumerate(current.runs) if item.id == run.id)
                current.runs[index] = snapshot

            await finish_write(self.store.update, project.id, save)

    async def stage(self, project: Project, run: Run, stage: str) -> None:
        run.stage = stage
        await self.checkpoint(project, run)

    async def execute(self, project: Project, run: Run) -> None:
        usage = RunUsage()
        usage_token = run_usage.set(usage)
        mask_token = document_masks.set(DocumentMasks())
        try:
            try:
                if run.kind == 'extract':
                    await self.extract(project, run)
                else:
                    await self.generate(project, run)
            except asyncio.CancelledError:
                if run.status != 'completed':
                    run.status, run.issues = 'interrupted', ['処理が中断しました。再実行できます。']
            except Exception as error:
                run.status, run.issues = 'failed', [self.error_message(error)]
            await self.checkpoint(project, run)
        finally:
            document_masks.reset(mask_token)
            run_usage.reset(usage_token)
            if run.status == 'running':
                run.status = 'interrupted'
            usage.emit(run.id, run.kind, run.status, run.attempt)

    @staticmethod
    def error_message(error: Exception) -> str:
        if isinstance(error, ServiceError):
            return str(error)
        return f'処理に失敗しました（{type(error).__name__}）。入力・設定を確認して再実行してください。'

    async def extract(self, project: Project, run: Run) -> None:
        await self.stage(project, run, 'extracting')
        data = await self.ai.extraction_data(project)
        draft = await self.ai.plan_extraction(data)
        if draft.searches:
            draft = await self.refine_extraction(project, run, data, draft)
        run.changes = [Change.model_validate({**item.model_dump(exclude={'evidence_ids'}), 'id': identifier()})
                       for item in draft.changes]
        revision = project.revision + 1

        def apply(current):
            current.changes, current.revision = run.changes, revision
            saved = get_run(current, run.id)
            saved.changes, saved.revision = current.changes, current.revision
            saved.status, saved.stage = 'completed', 'awaiting_confirmation'
            saved.extraction_warnings = run.extraction_warnings

        try:
            await finish_write(self.store.update, project.id, apply)
        except asyncio.CancelledError:
            run.revision = revision
            run.status, run.stage = 'completed', 'awaiting_confirmation'
            raise
        run.revision = revision
        run.status, run.stage = 'completed', 'awaiting_confirmation'

    async def refinement_candidates(self, searches, seen_queries: set[str], warnings: list[str]):
        queries = 0
        results = []
        for item in searches:
            key = unicodedata.normalize('NFKC', ' '.join(item.query.split())).casefold()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries += 1
            try:
                query = await self.ai.search_query(item)
                if not query:
                    warnings.append('マスクで検出した語や伏せ字を含むなど、安全に送信できない検索語がありました。')
                else:
                    documents = await self.kb.search_query(query)
                    if not documents:
                        warnings.append('検索に一致する既存仕様が見つかりませんでした。')
                    results.append([(candidate, query) for candidate in documents])
            except (ServiceError, ValueError):
                warnings.append('仕様検索または検索語のマスクに失敗しました。')
            if queries == MAX_QUERIES_PER_ROUND:
                break
        candidates = {}
        for index in range(max(map(len, results), default=0)):
            for result in results:
                if index < len(result):
                    candidate, query = result[index]
                    candidates.setdefault(candidate.document_id, (candidate, query))
        return candidates

    async def refinement_downloads(self, candidates, attempted: set[str], warnings: list[str]):
        evidence = []
        limit = min(self.settings.refinement_max_documents, len(attempted) +
                    self.settings.refinement_documents_per_round)
        for document_id, (candidate, query) in candidates.items():
            if len(attempted) >= limit:
                break
            if document_id in attempted:
                continue
            attempted.add(document_id)
            try:
                async with self.semaphore:
                    document = await self.kb.download(candidate)
            except (ServiceError, ValueError):
                warnings.append('取得できなかった仕様書があります。')
                continue
            try:
                await self.ai.mask_document(document)
            except ServiceError:
                warnings.append('マスクに失敗した仕様書は推敲に使用しませんでした。')
                continue
            evidence.append((f'kb_{len(attempted)}', document, query))
        return evidence

    async def refine_extraction(self, project: Project, run: Run, data, draft):
        seen_queries, attempted = set(), set()
        evidence, warnings = [], []
        for _ in range(self.settings.refinement_max_rounds):
            await self.stage(project, run, 'searching_requirements')
            candidates = await self.refinement_candidates(draft.searches, seen_queries, warnings)
            found = await self.refinement_downloads(candidates, attempted, warnings)
            if not found:
                warnings.append('追加で確認できる既存仕様がありませんでした。')
                break
            evidence.extend(found)
            await self.stage(project, run, 'refining_requirements')
            try:
                refined = await self.ai.refine(data, draft, evidence)
            except ServiceError:
                warnings.append('仕様のマスクまたは推敲に失敗しました。最後に得られた整理結果を表示します。')
                break
            grounded = refined.reviewed_evidence_ids or any(change.evidence_ids for change in refined.changes)
            if not grounded:
                if not refined.searches:
                    warnings.append('確認事項の根拠となる仕様の記載を特定できませんでした。')
                    break
                draft = draft.model_copy(update={'searches': refined.searches})
            else:
                draft = refined
                if not draft.searches:
                    break
            if len(attempted) >= self.settings.refinement_max_documents:
                warnings.append('仕様書の取得上限に達し、追加の確認ができませんでした。')
                break
        else:
            warnings.append('検索と推敲の上限に達し、追加の確認ができませんでした。')
        if draft.searches:
            warnings.extend(f'未確認：{item.question}' for item in draft.searches)
        run.extraction_warnings = list(dict.fromkeys(warnings))
        return draft

    async def generate(self, project: Project, run: Run) -> None:
        for change in run.changes:
            await self.evaluate_change(project, run, change)
        groups = defaultdict(list)
        for change in run.changes:
            if change.id in run.selections:
                groups[run.selections[change.id]].append(change)
        for document_id, changes in groups.items():
            previous = run.proposals.get(document_id)
            if previous is None or set(previous.change_ids) != {change.id for change in changes}:
                run.proposals.pop(document_id, None)
                await self.propose(project, run, document_id, changes)
        run.status, run.stage = ('review' if run.issues else 'completed'), 'finished'

    async def evaluate_change(self, project: Project, run: Run, change) -> None:
        if change.id in run.selections:
            return
        try:
            await self.stage(project, run, 'searching')
            if change.id not in run.candidates:
                run.candidates[change.id] = await self.kb.search(change)
                await self.checkpoint(project, run)
            await self.evaluate_candidates(project, run, change)
        except Exception as error:
            run.issues.append(f'{change.summary}: {self.error_message(error)}')
            await self.checkpoint(project, run)

    async def evaluate_candidates(self, project: Project, run: Run, change) -> None:
        candidates = run.candidates[change.id]
        run.assessments.setdefault(change.id, {})
        await self.stage(project, run, 'assessing')
        results = await asyncio.gather(*(self.evaluate_one(project, run, change, item) for item in candidates),
                                       return_exceptions=True)
        if any(isinstance(result, BaseException) for result in results):
            failures = [self.error_message(result) for result in results if isinstance(result, Exception)]
            raise ServiceError('候補の評価が不完了です。' + ' / '.join(dict.fromkeys(failures)))
        selected = choose(candidates, run.assessments[change.id], self.settings.necessity_threshold)
        if selected:
            run.selections[change.id] = selected
        else:
            run.issues.append(f'{change.summary}: 候補なし、または必要性がしきい値を超えません。入力や検索範囲を確認してください。')
        await self.checkpoint(project, run)

    async def evaluate_one(self, project: Project, run: Run, change, candidate) -> None:
        async with self.semaphore:
            document_id = candidate.document_id
            if document_id not in run.documents:
                run.documents[document_id] = await self.kb.download(candidate)
                await self.checkpoint(project, run)
            if document_id not in run.assessments[change.id]:
                assessment = await self.ai.assess(change, run.documents[document_id])
                run.assessments[change.id][document_id] = assessment
                await self.checkpoint(project, run)

    async def propose(self, project: Project, run: Run, document_id: str, changes) -> None:
        await self.stage(project, run, 'generating')
        document = run.documents[document_id]
        try:
            generated = await self.ai.generate(changes, document)
            expected = {change.id for change in changes}
            if set(generated.change_ids) != expected or len(generated.change_ids) != len(expected):
                raise ServiceError('修正案の変更IDに欠落・重複があります。')
            if generated.modified == document.text:
                raise ServiceError('修正案が原文と同一です。')
            run.proposals[document_id] = Proposal(**generated.model_dump(), document_id=document_id,
                                                  file_path=document.file_path, original=document.text,
                                                  diff=markdown_diff(document.text, generated.modified,
                                                                     document.file_path))
        except Exception as error:
            run.issues.append(f'{document.file_path}: {self.error_message(error)}')
        await self.checkpoint(project, run)

    async def retry(self, run_id: str) -> Run:
        project, previous = await asyncio.to_thread(self.store.find_run, run_id)

        def reset(current):
            editable(current)
            run = get_run(current, run_id)
            if run.status not in {'failed', 'review', 'interrupted'} or run.revision != current.revision:
                raise Conflict('この実行は再開できません。最新の変更から新しく実行してください。')
            run.status, run.issues = 'running', []
            run.attempt += 1
            run.extraction_warnings = []
            run.candidates = {key: value for key, value in run.candidates.items() if value}

        project = await asyncio.to_thread(self.store.update, project.id, reset)
        run = get_run(project, previous.id)
        self.launch(project, run)
        return run

    async def close(self) -> None:
        pending = list(self.tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.ai.close()
