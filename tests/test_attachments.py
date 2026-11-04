from io import BytesIO

from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pypdf import PdfWriter
import pytest

from aadd.attachments import extract
from aadd.config import Settings


def binary(document):
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_text_and_safe_name():
    result = extract('../../input.md', '変更内容'.encode(), Settings())
    assert result.filename == 'input.md'
    assert result.text == '変更内容'


def test_word_tables_and_sources():
    document = Document()
    document.add_paragraph('変更要求')
    document.add_table(rows=1, cols=1).cell(0, 0).text = '変更後'
    result = extract('input.docx', binary(document), Settings())
    assert '段落 1: 変更要求' in result.text
    assert '表 1 行 1: 変更後' in result.text


def test_excel_formula_is_not_calculated():
    workbook = Workbook()
    workbook.active['A1'] = '=1+2'
    result = extract('input.xlsx', binary(workbook), Settings())
    assert 'セル A1: 数式==1+2; 保存済み計算値=未保存' in result.text


def test_powerpoint():
    presentation = Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[1]).shapes.title.text = '変更要求'
    assert 'スライド 1: 変更要求' in extract('input.pptx', binary(presentation), Settings()).text


@pytest.mark.parametrize('name,data', [('a.doc', b'data'), ('a.pdf', b'bad'), ('a.txt', b'')])
def test_invalid_files(name, data):
    with pytest.raises(ValueError, match=name):
        extract(name, data, Settings())


def test_size_and_archive_limits():
    with pytest.raises(ValueError):
        extract('a.txt', b'123', Settings(max_file_bytes=2))
    with pytest.raises(ValueError):
        extract('a.docx', binary(Document()), Settings(max_expanded_bytes=1))


def test_sparse_spreadsheet_limit():
    workbook = Workbook()
    workbook.active['XFD1048576'] = '巨大な範囲'
    with pytest.raises(ValueError):
        extract('huge.xlsx', binary(workbook), Settings(max_spreadsheet_cells=1000))


@pytest.mark.parametrize('encrypted', [False, True])
def test_empty_or_encrypted_pdf(encrypted):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    if encrypted:
        writer.encrypt('test')
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match='input.pdf'):
        extract('input.pdf', buffer.getvalue(), Settings())


def test_wrong_office_type_and_bad_encoding():
    with pytest.raises(ValueError):
        extract('wrong.docx', binary(Workbook()), Settings())
    with pytest.raises(ValueError):
        extract('bad.txt', b'\xff\xfe', Settings())
