from inference.tests.conftest import API_TEST_PASSWORD, API_TEST_USERNAME, REQUIRES_ESPEAK


def _login(api_client):
    api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": API_TEST_PASSWORD})


def test_create_batch_requires_auth(api_client):
    resp = api_client.post("/batches", files={"manifest": ("m.csv", "filename\ncall1.wav\n")})
    assert resp.status_code == 401


def test_list_batches_requires_auth(api_client):
    assert api_client.get("/batches").status_code == 401


def test_create_batch_rejects_manifest_without_filename_column(api_client):
    _login(api_client)
    resp = api_client.post("/batches", files={"manifest": ("m.csv", "not_filename\nfoo\n", "text/csv")})
    assert resp.status_code == 400


def test_create_batch_rejects_manifest_referencing_missing_file(api_client):
    _login(api_client)
    resp = api_client.post("/batches", files={"manifest": ("m.csv", "filename\nmissing.wav\n", "text/csv")})
    assert resp.status_code == 400
    assert "missing.wav" in resp.json()["detail"]


@REQUIRES_ESPEAK
def test_full_batch_upload_and_processing(api_client, clean_call):
    _login(api_client)

    with open(clean_call, "rb") as f:
        audio_bytes = f.read()

    resp = api_client.post(
        "/batches",
        files={
            "manifest": ("m.csv", "filename\ncall.wav\n", "text/csv"),
            "files": ("call.wav", audio_bytes, "audio/wav"),
        },
    )
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]
    assert resp.json()["total_calls"] == 1

    detail = api_client.get(f"/batches/{batch_id}").json()
    assert detail["batch"]["status"] == "done"
    assert detail["batch"]["completed_calls"] == 1
    assert len(detail["calls"]) == 1
    call = detail["calls"][0]
    assert call["status"] == "done"
    assert call["result"] is not None
    assert "emotional_tone" in call["result"]
    assert 0.0 <= call["result"]["confidence"] <= 1.0

    listing = api_client.get("/batches").json()
    assert any(b["id"] == batch_id for b in listing)

    csv_resp = api_client.get(f"/batches/{batch_id}/export?format=csv")
    assert csv_resp.status_code == 200
    assert "emotional_tone" in csv_resp.text
    assert "call.wav" in csv_resp.text

    delete_resp = api_client.delete(f"/batches/{batch_id}")
    assert delete_resp.status_code == 200
    assert api_client.get(f"/batches/{batch_id}").status_code == 404
    assert not any(b["id"] == batch_id for b in api_client.get("/batches").json())


def test_delete_batch_requires_auth(api_client):
    assert api_client.delete("/batches/nonexistent").status_code == 401


def test_delete_nonexistent_batch_is_404(api_client):
    _login(api_client)
    assert api_client.delete("/batches/nonexistent").status_code == 404
