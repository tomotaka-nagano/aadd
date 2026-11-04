from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
from pptx import Presentation

from aadd.config import Settings
from aadd.models import Attachment


OFFICE_PARTS = {'.docx': 'word/document.xml', '.xlsx': 'xl/workbook.xml', '.pptx': 'ppt/presentation.xml'}
FORMATS = {'.txt', '.md', '.pdf', *OFFICE_PARTS}


def validate_archive(data: bytes, suffix: str, settings: Settings) -> None:
    with ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > settings.max_expanded_bytes:
            raise ValueError('展開後のファイルが上限を超えています。')
        if OFFICE_PARTS[suffix] not in archive.namelist():
            raise ValueError('拡張子とファイル形式が一致しません。')
        if any('vbaproject' in item.filename.lower() for item in entries):
            raise ValueError('マクロを含む資料は扱えません。')


def word_text(data: bytes) -> list[str]:
    document = Document(BytesIO(data))
    paragraphs = [f'段落 {index}: {paragraph.text}' for index, paragraph in enumerate(document.paragraphs, 1)
                  if paragraph.text.strip()]
    for index, table in enumerate(document.tables, 1):
        paragraphs.extend(f'表 {index} 行 {row_id}: ' + ' | '.join(cell.text for cell in row.cells)
                          for row_id, row in enumerate(table.rows, 1))
    return paragraphs


def excel_text(data: bytes, cell_limit: int):
    formulas = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
    values = load_workbook(BytesIO(data), read_only=True, data_only=True, keep_links=False)
    try:
        if sum((sheet.max_row or 0) * (sheet.max_column or 0) for sheet in formulas) > cell_limit:
            raise ValueError('Excelの走査セル数が上限を超えています。')
        for sheet in formulas:
            for formula_row, value_row in zip(sheet.iter_rows(), values[sheet.title].iter_rows(), strict=True):
                for cell, cached in zip(formula_row, value_row, strict=True):
                    if cell.value is not None:
                        yield excel_cell(sheet.title, cell, cached.value)
    finally:
        formulas.close()
        values.close()


def excel_cell(sheet: str, cell, cached) -> str:
    value = str(cell.value)
    if cell.data_type == 'f':
        value = f'数式={cell.value}; 保存済み計算値={cached if cached is not None else "未保存"}'
    return f'シート {sheet} セル {cell.coordinate}: {value}'


def slide_text(data: bytes) -> list[str]:
    lines = []
    for index, slide in enumerate(Presentation(BytesIO(data)).slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.strip():
                lines.append(f'スライド {index}: {shape.text}')
            if shape.has_table:
                lines.extend(f'スライド {index} 表: ' + ' | '.join(cell.text for cell in row.cells)
                             for row in shape.table.rows)
    return lines


def pdf_text(data: bytes) -> list[str]:
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise ValueError('暗号化PDFは扱えません。')
    return [f'ページ {index}: {text}' for index, page in enumerate(reader.pages, 1)
            if (text := page.extract_text() or '').strip()]


def bounded_text(lines, limit: int) -> str:
    collected, length = [], 0
    for line in lines:
        length += len(line) + 1
        if length > limit:
            raise ValueError('抽出文字数が上限を超えています。')
        collected.append(line)
    return '\n'.join(collected)


def extract(filename: str, data: bytes, settings: Settings) -> Attachment:
    name = PurePosixPath(filename.replace('\\', '/')).name
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in FORMATS or not data or len(data) > settings.max_file_bytes:
        raise ValueError(f'{name}: 非対応形式、空ファイル、またはサイズ上限超過です。')
    try:
        return parse_attachment(name, suffix, data, settings)
    except Exception as error:
        raise ValueError(f'{name}: 読取りに失敗しました。形式・暗号化・内容を確認してください。') from error


def parse_attachment(name: str, suffix: str, data: bytes, settings: Settings) -> Attachment:
    if suffix in OFFICE_PARTS:
        validate_archive(data, suffix, settings)
    parsers = {'.docx': word_text, '.xlsx': lambda value: excel_text(value, settings.max_spreadsheet_cells),
               '.pptx': slide_text, '.pdf': pdf_text}
    text = (bounded_text(parsers[suffix](data), settings.max_input_chars)
            if suffix in parsers else data.decode('utf-8-sig'))
    if not text.strip() or len(text) > settings.max_input_chars:
        raise ValueError('抽出結果が空、または文字数上限を超えています。')
    warnings = ['画像・図形・埋め込み資料は解析対象外です。'] if suffix in parsers else []
    return Attachment(filename=name, text=text, warnings=warnings)
