import pytest

from aadd.models import Project, Run
from aadd.storage import Conflict, Store


def test_persistence_and_restart(tmp_path):
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='request', runs=[Run(kind='extract', revision=0)]))
    restarted = Store(tmp_path)
    restarted.interrupt()
    assert restarted.get(project.id).runs[0].status == 'interrupted'
    assert restarted.find_run(project.runs[0].id)[0].id == project.id


def test_running_project_cannot_be_deleted(tmp_path):
    store = Store(tmp_path)
    project = store.create(Project(title='Test', request='request', runs=[Run(kind='extract', revision=0)]))
    with pytest.raises(Conflict):
        store.delete(project.id)
