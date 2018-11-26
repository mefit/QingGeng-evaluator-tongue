import pytest
import tongue

@pytest.fixture
def client():
    client = tongue.app.test_client()
    yield client

def test_listen(client):
    rv = client.post('/')
    assert '200 OK' == rv.status
    assert b'ok' == rv.data

