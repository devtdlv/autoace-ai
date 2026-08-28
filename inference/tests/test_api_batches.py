import io
import json
import zipfile

from inference.tests.conftest import API_TEST_PASSWORD, API_TEST_USERNAME, REQUIRES_ESPEAK


def _login(api_client):
    api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": API_TEST_PASSWORD})


def test_create_batch_requires_auth(api_client):
    resp = api_client.post("/batches", files={"manifest": ("m.csv", "name\ncall1.wav\n")})
    assert resp.status_code == 401


def test_list_batches_requires_auth(api_client):
    assert api_client.get("/batches").status_code == 401


def test_create_batch_rejects_no_audio_files(api_client):
    _login(api_client)
    resp = api_client.post("/batches", files={"manifest": ("m.csv", "name\nx.wav\n", "text/csv")})
    assert resp.status_code == 400
    assert "No audio files" in resp.json()["detail"]


def test_create_batch_rejects_manifest_without_name_column(api_client):
    _login(api_client)
    resp = api_client.post(
        "/batches",
        files={
            "manifest": ("m.csv", "not_a_name_column\nfoo\n", "text/csv"),
            "files": ("foo.wav", b"fake audio bytes", "audio/wav"),
        },
    )
    assert resp.status_code == 400
    assert "'name' column" in resp.json()["detail"]


def test_create_batch_rejects_manifest_referencing_missing_file(api_client):
    _login(api_client)
    resp = api_client.post(
        "/batches",
        files={
            "manifest": ("m.csv", "name\nmissing.wav\n", "text/csv"),
            "files": ("present.wav", b"fake audio bytes", "audio/wav"),
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "missing.wav" in detail
    assert "not uploaded" in detail


def test_create_batch_rejects_uploaded_file_not_in_manifest(api_client):
    _login(api_client)
    resp = api_client.post(
        "/batches",
        files=[
            ("manifest", ("m.csv", "name\nexpected.wav\n", "text/csv")),
            ("files", ("expected.wav", b"fake audio bytes", "audio/wav")),
            ("files", ("extra.wav", b"fake audio bytes", "audio/wav")),
        ],
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "extra.wav" in detail
    assert "not listed in the manifest" in detail


@REQUIRES_ESPEAK
def test_create_batch_without_manifest_processes_all_uploaded_files(api_client, clean_call):
    _login(api_client)
    with open(clean_call, "rb") as f:
        audio_bytes = f.read()

    resp = api_client.post("/batches", files={"files": ("call.wav", audio_bytes, "audio/wav")})
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]
    assert resp.json()["total_calls"] == 1

    detail = api_client.get(f"/batches/{batch_id}").json()
    assert detail["batch"]["manifest_name"] == "1 file(s), no manifest"
    assert detail["calls"][0]["filename"] == "call.wav"


@REQUIRES_ESPEAK
def test_create_batch_ignores_macos_zip_metadata_entries(api_client, clean_call):
    """A ZIP built by macOS Archive Utility includes __MACOSX/._call.wav
    AppleDouble entries alongside real files — these must not be treated
    as audio or trip the 'uploaded but not in manifest' validation check.
    """
    _login(api_client)
    with open(clean_call, "rb") as f:
        audio_bytes = f.read()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("call.wav", audio_bytes)
        zf.writestr("__MACOSX/._call.wav", b"fake resource fork data")
        zf.writestr("labels.csv", "name\ncall.wav\n")
    zip_buffer.seek(0)

    resp = api_client.post(
        "/batches",
        files={"archive": ("evaluation_batch.zip", zip_buffer.read(), "application/zip")},
    )
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]
    assert resp.json()["total_calls"] == 1

    detail = api_client.get(f"/batches/{batch_id}").json()
    assert len(detail["calls"]) == 1
    assert detail["calls"][0]["filename"] == "call.wav"


@REQUIRES_ESPEAK
def test_create_batch_auto_detects_manifest_embedded_in_files(api_client, clean_call):
    """Matches the spec's single-upload shape: audio + one CSV manifest
    selected together, not as a separate upload field."""
    _login(api_client)
    with open(clean_call, "rb") as f:
        audio_bytes = f.read()

    resp = api_client.post(
        "/batches",
        files=[
            ("files", ("call.wav", audio_bytes, "audio/wav")),
            ("files", ("manifest.csv", "name\ncall.wav\n", "text/csv")),
        ],
    )
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]
    assert resp.json()["total_calls"] == 1

    detail = api_client.get(f"/batches/{batch_id}").json()
    assert detail["batch"]["manifest_name"] == "manifest.csv"
    assert len(detail["calls"]) == 1
    assert detail["calls"][0]["filename"] == "call.wav"


@REQUIRES_ESPEAK
def test_full_batch_upload_and_processing(api_client, clean_call):
    _login(api_client)

    with open(clean_call, "rb") as f:
        audio_bytes = f.read()

    expected = {"emotional_tone": "frustrated", "confidence": 0.9}
    manifest_csv = 'name,result_json\ncall.wav,"' + json.dumps(expected).replace('"', '""') + '"\n'

    resp = api_client.post(
        "/batches",
        files={
            "manifest": ("m.csv", manifest_csv, "text/csv"),
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
    assert call["expected"] == expected

    listing = api_client.get("/batches").json()
    assert any(b["id"] == batch_id for b in listing)

    csv_resp = api_client.get(f"/batches/{batch_id}/export?format=csv")
    assert csv_resp.status_code == 200
    assert "emotional_tone" in csv_resp.text
    assert "call.wav" in csv_resp.text
    assert "expected_json" in csv_resp.text

    delete_resp = api_client.delete(f"/batches/{batch_id}")
    assert delete_resp.status_code == 200
    assert api_client.get(f"/batches/{batch_id}").status_code == 404
    assert not any(b["id"] == batch_id for b in api_client.get("/batches").json())


def test_corrupt_file_in_batch_gets_a_valid_fallback_result_not_a_gap(api_client):
    """A single unprocessable file must not sink the batch, and — per the
    hidden-set robustness fix — should still leave a schema-valid,
    confidence=0.0 guess behind rather than an empty/missing result.
    """
    _login(api_client)
    resp = api_client.post(
        "/batches",
        files={"files": ("corrupt.wav", b"not actually audio data at all", "audio/wav")},
    )
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]

    detail = api_client.get(f"/batches/{batch_id}").json()
    assert detail["batch"]["status"] == "done"
    call = detail["calls"][0]
    assert call["status"] == "failed"
    assert call["error"]
    assert call["result"] is not None
    assert call["result"]["confidence"] == 0.0
    assert call["result"]["emotional_tone"] == "neutral"


def test_delete_batch_requires_auth(api_client):
    assert api_client.delete("/batches/nonexistent").status_code == 401


def test_delete_nonexistent_batch_is_404(api_client):
    _login(api_client)
    assert api_client.delete("/batches/nonexistent").status_code == 404
