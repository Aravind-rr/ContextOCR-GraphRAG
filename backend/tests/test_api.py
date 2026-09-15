from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_status():
    assert client.get("/health").json() == {"status":"ok"}
    response = client.get("/api/system/status")
    assert response.status_code == 200
    body = response.json()
    assert body["components"]["backend"]["status"] == "READY"
    assert "totals" in body


def test_upload_rejects_bad_type():
    response=client.post("/api/documents/upload",files={"file":("bad.exe",b"not a document","application/octet-stream")})
    assert response.status_code == 415

