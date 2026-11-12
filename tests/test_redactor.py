import asyncio
import json

import httpx
import pytest

from aadd.config import Settings
from aadd.external import ServiceError
from aadd.redactor import Redactor, replace_strings, strings


def test_string_replacement_preserves_json_shape():
    data = {'items': ['秘密社', {'name': '秘密社', 'score': 0.8}], 'empty': None, 'flag': True}
    assert list(strings(data)) == ['秘密社', '秘密社']
    assert replace_strings(data, iter(['[社1]', '[社1]'])) == {
        'items': ['[社1]', {'name': '[社1]', 'score': 0.8}], 'empty': None, 'flag': True,
    }


async def test_mask_request_options_and_mapping_are_preserved():
    mapping = [{'placeholder': '[社1]', 'canonical': '秘密社', 'extra_field': {'preserved': True}}]

    def handler(request):
        assert str(request.url) == 'http://redactor.test/prefix/mask'
        assert request.headers['Authorization'] == 'Bearer test'
        body = json.loads(request.content)
        assert body == {'parts': [{'id': '0', 'text': '秘密社'}, {'id': '1', 'text': '秘密社'}],
                        'detection': 'ner', 'mask_level': 'weak'}
        return httpx.Response(200, json={
            'masked_parts': [{'id': '0', 'masked_text': '[社1]'}, {'id': '1', 'masked_text': '[社1]'}],
            'mapping': mapping,
        })

    settings = Settings(redactor_url='http://redactor.test/prefix/', redactor_headers={'Authorization': 'Bearer test'},
                        redactor_detection='ner', redactor_mask_level='weak')
    result = await Redactor(settings, httpx.MockTransport(handler)).mask(['秘密社', '秘密社'])
    assert result.mapping == mapping
    assert [part.masked_text for part in result.masked_parts] == ['[社1]', '[社1]']


@pytest.mark.parametrize('body', [
    {}, {'masked_parts': [], 'mapping': []},
    {'masked_parts': [{'id': '1', 'masked_text': 'x'}], 'mapping': []},
    {'masked_parts': [{'id': '0', 'masked_text': None}], 'mapping': []},
    {'masked_parts': [{'id': '0', 'masked_text': 123}], 'mapping': []},
    {'masked_parts': [{'id': '0', 'masked_text': 'x'}]},
    {'masked_parts': [{'id': '0', 'masked_text': 'x'}, {'id': '0', 'masked_text': 'y'}], 'mapping': []},
])
async def test_invalid_mask_response_fails_closed(body):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    redactor = Redactor(Settings(retry_count=0), transport)
    with pytest.raises(ServiceError, match='マスクAPI'):
        await redactor.mask(['秘密社'])


@pytest.mark.parametrize('body', [{}, {'restored_text': None}, {'restored_text': 123}])
async def test_invalid_unmask_response_is_not_a_success(body):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ServiceError, match='マスクAPI'):
        await Redactor(Settings(retry_count=0), transport).unmask({'name': '[社1]'}, [])


async def test_transient_mask_failure_is_retried():
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={'masked_parts': [{'id': '0', 'masked_text': '[社1]'}], 'mapping': []})

    result = await Redactor(Settings(retry_count=1), httpx.MockTransport(handler)).mask(['秘密社'])
    assert result.masked_parts[0].masked_text == '[社1]'
    assert len(attempts) == 2


async def test_timeout_is_an_explicit_failure():
    def handler(request):
        raise httpx.ReadTimeout('秘密社', request=request)

    with pytest.raises(ServiceError, match='マスクAPI') as error:
        await Redactor(Settings(retry_count=0), httpx.MockTransport(handler)).mask(['秘密社'])
    assert '秘密社' not in str(error.value)


async def test_concurrent_calls_keep_their_own_mapping():
    async def handler(request):
        body = json.loads(request.content)
        await asyncio.sleep(0)
        if request.url.path == '/mask':
            return httpx.Response(200, json={
                'masked_parts': [{'id': '0', 'masked_text': '[社1]'}],
                'mapping': [{'placeholder': '[社1]', 'canonical': body['parts'][0]['text']}],
            })
        return httpx.Response(200, json={'restored_text': body['mapping'][0]['canonical']})

    redactor = Redactor(Settings(), httpx.MockTransport(handler))

    async def roundtrip(text):
        result = await redactor.mask([text])
        return await redactor.unmask({'name': result.masked_parts[0].masked_text}, result.mapping)

    assert await asyncio.gather(roundtrip('秘密A社'), roundtrip('秘密B社')) == [
        {'name': '秘密A社'}, {'name': '秘密B社'},
    ]
