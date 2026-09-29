import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import main
import summarizer


def test_modes_keep_original_and_prompt():
    original = '前半\n後半🙂'
    assert summarizer.process_transcript(original, 'transcript', 'ja') == original
    assert original in summarizer.process_transcript(original, 'prompt', 'ja')
    with pytest.raises(ValueError):
        summarizer.chunk_text('abc', 2, 2)


def test_long_summary_includes_tail_and_reduces(monkeypatch):
    calls = []
    def post(self, url, **kwargs):
        text = kwargs['json']['messages'][1]['content']
        calls.append(text)
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':'部分要約: '+str(len(calls))}}]})
    monkeypatch.setattr(httpx.Client, 'post', post)
    text = '開始' + '中間資料。' * 9000 + '最後の重要事実'
    result = summarizer.process_transcript(text,'summary','ja',groq_api_key='test-key')
    assert '最後の重要事実' in ''.join(calls[:-1])
    assert len(calls) > 2
    assert all('部分要約: '+str(n) in calls[-1] for n in range(1,len(calls)))
    assert result.startswith('部分要約')


def test_key_missing_and_api_errors_are_honest(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    with pytest.raises(ValueError,match='APIキー'):
        summarizer.process_transcript('原文','summary','ja')
    monkeypatch.setattr(httpx.Client,'post',lambda *a,**k:httpx.Response(429,json={'error':'secret must not leak'}))
    with pytest.raises(ValueError,match='利用上限') as error:
        summarizer.process_transcript('原文','summary','ja',groq_api_key='private')
    assert 'secret' not in str(error.value)


def test_truncated_model_output_is_not_success(monkeypatch):
    monkeypatch.setattr(httpx.Client,'post',lambda *a,**k:httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}))
    with pytest.raises(ValueError,match='途中'):
        summarizer.process_transcript('原文','summary','ja',groq_api_key='test')


def events(response):
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:')]


def test_sse_keeps_transcript_on_summary_failure(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    with patch('transcriber.get_transcript',return_value={'text':'原本の全文','method':'captions'}):
        response=TestClient(main.app).post('/process',json={'url':'https://youtu.be/jNQXAC9IVRw','mode':'summary','language':'ja'})
    result=events(response)
    assert next(e['text'] for e in result if e['type']=='transcript')=='原本の全文'
    assert any(e['type']=='error' for e in result)
    assert not any(e['type']=='result' for e in result)


def test_audio_original_and_summary_both_return():
    with patch('transcriber.get_transcript_from_file',return_value={'text':'音声原本','method':'groq'}), patch('summarizer.process_transcript',return_value='音声の要約'):
        response=TestClient(main.app).post('/process-file',data={'mode':'summary','language':'ja','groq_api_key':'test'},files={'file':('sample.wav',b'fixture','audio/wav')})
    result=events(response)
    assert next(e['text'] for e in result if e['type']=='transcript')=='音声原本'
    assert next(e['text'] for e in result if e['type']=='result')=='音声の要約'
