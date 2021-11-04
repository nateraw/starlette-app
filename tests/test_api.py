import pytest
from starlette.testclient import TestClient
import os
from pathlib import Path


@pytest.fixture(params=['nateraw/rare-puppers', 'nateraw/asdf'])
def app(request):
    param = request.param
    os.environ['TASK'] = 'image-classification'
    os.environ['MODEL_ID'] = param
    from app.main import app, get_pipeline
    get_pipeline.cache_clear()
    yield app
    del os.environ['TASK']
    del os.environ['MODEL_ID']


@pytest.fixture(params=['plane1.jpg', 'plane2.jpg'])
def payload(request):
    filename = request.param
    filepath = Path(__file__).parent / filename
    with filepath.open('rb') as f:
        data = f.read()
    return data


def test_api_get(app):
    with TestClient(app) as client:
        response = client.get('/')
    assert response.status_code == 200


def test_api_post(app, payload):
    with TestClient(app) as client:
        response = client.post('/', data=payload)
    assert response.status_code == 200
