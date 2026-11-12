import asyncio
from pathlib import Path
import shutil
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile

from aadd.attachments import extract
from aadd.models import ChangeUpdate, MaskPreview, Project, RunRequest
from aadd.storage import Conflict, editable
from aadd.workflow import update_changes


router = APIRouter(prefix='/api')


def services(request: Request):
    return request.app.state.store, request.app.state.workflow


@router.get('/health')
def health(request: Request):
    return {'status': 'ok', 'ai_configured': request.app.state.settings.ai_config.is_file()}


@router.get('/projects')
def projects(request: Request):
    store, _ = services(request)
    return [{'id': item.id, 'title': item.title, 'created_at': item.created_at, 'revision': item.revision,
             'status': item.runs[-1].status if item.runs else 'draft'} for item in store.list()]


def save_uploads(directory: Path, project: Project, originals: list[bytes]) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for attachment, data in zip(project.attachments, originals, strict=True):
            (directory / attachment.id).write_bytes(data)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def read_project(request: Request) -> tuple[Project, list[bytes]]:
    settings = request.app.state.settings
    async with request.form(max_files=settings.max_files, max_fields=3,
                            max_part_size=settings.max_input_chars * 4) as form:
        text = str(form.get('request', '')).strip()
        files = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
        if len(text) > settings.max_input_chars or len(files) > settings.max_files:
            raise HTTPException(413, '入力が上限を超えています。')
        attachments, originals = await read_attachments(files, settings)
    if not text and not attachments:
        raise ValueError('変更要求または添付資料を入力してください。')
    if len(text) + sum(len(item.text) for item in attachments) > settings.max_input_chars:
        raise HTTPException(413, '入力と添付の抽出文字数の合計が上限を超えています。')
    title = text.splitlines()[0][:80] if text else attachments[0].filename
    return Project(title=title, request=text, attachments=attachments), originals


async def read_attachments(files, settings):
    attachments, originals = [], []
    for file in files:
        data = await file.read(settings.max_file_bytes + 1)
        attachments.append(await asyncio.to_thread(extract, file.filename or 'unnamed', data, settings))
        originals.append(data)
    return attachments, originals


@router.post('/projects', status_code=201)
async def create_project(request: Request):
    store, _ = services(request)
    project, originals = await read_project(request)
    directory = request.app.state.settings.data_dir / 'attachments' / project.id
    await asyncio.to_thread(save_uploads, directory, project, originals)
    try:
        return await asyncio.to_thread(store.create, project)
    except Exception:
        await asyncio.to_thread(shutil.rmtree, directory, True)
        raise


@router.post('/extraction-previews', response_model=MaskPreview)
async def preview_new_extraction(request: Request):
    project, _ = await read_project(request)
    return await services(request)[1].ai.preview_extraction(project)


@router.post('/projects/{project_id}/extraction-previews', response_model=MaskPreview)
async def preview_saved_extraction(project_id: str, body: RunRequest, request: Request):
    store, workflow = services(request)
    project = await asyncio.to_thread(store.get, project_id)
    editable(project)
    if project.revision != body.revision:
        raise Conflict('古い版です。案件を再読込みしてください。')
    return await workflow.ai.preview_extraction(project)


@router.get('/projects/{project_id}')
def project_detail(project_id: str, request: Request):
    return services(request)[0].get(project_id)


@router.delete('/projects/{project_id}', status_code=204)
def delete_project(project_id: str, request: Request):
    services(request)[0].delete(project_id)
    directory = request.app.state.settings.data_dir / 'attachments' / project_id
    shutil.rmtree(directory, ignore_errors=True)


@router.put('/projects/{project_id}/changes')
def save_changes(project_id: str, update: ChangeUpdate, request: Request):
    store, _ = services(request)
    return store.update(project_id, lambda item: update_changes(item, update))


@router.post('/projects/{project_id}/extractions', status_code=202)
async def start_extraction(project_id: str, body: RunRequest, request: Request):
    return await services(request)[1].start(project_id, 'extract', body.revision)


@router.post('/projects/{project_id}/runs', status_code=202)
async def start_generation(project_id: str, body: RunRequest, request: Request):
    return await services(request)[1].start(project_id, 'generate', body.revision)


@router.get('/runs/{run_id}')
def run_detail(run_id: str, request: Request):
    project, run = services(request)[0].find_run(run_id)
    return {'project_id': project.id, 'run': run, 'stale': run.revision != project.revision}


@router.post('/runs/{run_id}/retry', status_code=202)
async def retry(run_id: str, request: Request):
    return await services(request)[1].retry(run_id)


def find_proposal(store, project_id: str, proposal_id: str):
    project = store.get(project_id)
    for run in reversed(project.runs):
        for proposal in run.proposals.values():
            if proposal.id == proposal_id:
                return proposal
    raise KeyError('修正案が見つかりません。')


@router.get('/projects/{project_id}/proposals/{proposal_id}')
def proposal_detail(project_id: str, proposal_id: str, request: Request):
    return find_proposal(services(request)[0], project_id, proposal_id)


@router.get('/projects/{project_id}/proposals/{proposal_id}/download')
def download(project_id: str, proposal_id: str, request: Request, format: Literal['markdown', 'diff'] = 'markdown'):
    proposal = find_proposal(services(request)[0], project_id, proposal_id)
    content = proposal.modified if format == 'markdown' else proposal.diff
    extension = 'md' if format == 'markdown' else 'diff'
    return Response(content, media_type='text/plain; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="proposal-{proposal.id}.{extension}"'})
