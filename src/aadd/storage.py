import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from aadd.models import Project, Run


class Conflict(ValueError):
    pass


async def finish_write(operation, *arguments):
    task = asyncio.create_task(asyncio.to_thread(operation, *arguments))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def editable(project: Project) -> None:
    if any(run.status == 'running' for run in project.runs):
        raise Conflict('処理中は編集・削除・別の処理開始ができません。')


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'aadd.sqlite3'
        with self.connection() as database:
            database.execute('CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, body TEXT NOT NULL)')

    @contextmanager
    def connection(self):
        database = sqlite3.connect(self.path, timeout=30)
        try:
            with database:
                yield database
        finally:
            database.close()

    def create(self, project: Project) -> Project:
        with self.connection() as database:
            database.execute('INSERT INTO projects VALUES (?, ?)', (project.id, project.model_dump_json()))
        return project

    def get(self, project_id: str) -> Project:
        with self.connection() as database:
            row = database.execute('SELECT body FROM projects WHERE id=?', (project_id,)).fetchone()
        if row is None:
            raise KeyError('案件が見つかりません。')
        return Project.model_validate_json(row[0])

    def list(self) -> list[Project]:
        with self.connection() as database:
            rows = database.execute('SELECT body FROM projects ORDER BY rowid DESC').fetchall()
        return [Project.model_validate_json(row[0]) for row in rows]

    def update(self, project_id: str, mutation: Callable[[Project], None]) -> Project:
        with self.connection() as database:
            database.execute('BEGIN IMMEDIATE')
            row = database.execute('SELECT body FROM projects WHERE id=?', (project_id,)).fetchone()
            if row is None:
                raise KeyError('案件が見つかりません。')
            project = Project.model_validate_json(row[0])
            mutation(project)
            database.execute('UPDATE projects SET body=? WHERE id=?', (project.model_dump_json(), project_id))
        return project

    def delete(self, project_id: str) -> None:
        with self.connection() as database:
            database.execute('BEGIN IMMEDIATE')
            row = database.execute('SELECT body FROM projects WHERE id=?', (project_id,)).fetchone()
            if row is None:
                raise KeyError('案件が見つかりません。')
            editable(Project.model_validate_json(row[0]))
            database.execute('DELETE FROM projects WHERE id=?', (project_id,))

    def find_run(self, run_id: str) -> tuple[Project, Run]:
        for project in self.list():
            for run in project.runs:
                if run.id == run_id:
                    return project, run
        raise KeyError('実行が見つかりません。')

    def interrupt(self) -> None:
        for project in self.list():
            self.update(project.id, self._interrupt)

    @staticmethod
    def _interrupt(project: Project) -> None:
        for run in project.runs:
            if run.status == 'running':
                run.status = 'interrupted'
                run.issues = ['サーバー停止により中断しました。再実行できます。']
