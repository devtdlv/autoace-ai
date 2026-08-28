from inference.tests.conftest import API_TEST_PASSWORD, API_TEST_USERNAME


def test_login_with_correct_credentials_sets_session_cookie(api_client):
    resp = api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": API_TEST_PASSWORD})
    assert resp.status_code == 200
    assert "autoace_session" in resp.cookies


def test_login_with_wrong_password_is_rejected(api_client):
    resp = api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": "wrong"})
    assert resp.status_code == 401


def test_login_with_wrong_username_is_rejected(api_client):
    resp = api_client.post("/auth/login", json={"username": "nobody", "password": API_TEST_PASSWORD})
    assert resp.status_code == 401


def test_me_requires_session(api_client):
    resp = api_client.get("/auth/me")
    assert resp.status_code == 401


def test_me_returns_username_after_login(api_client):
    api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": API_TEST_PASSWORD})
    resp = api_client.get("/auth/me")
    assert resp.status_code == 200
    assert resp.json()["username"] == API_TEST_USERNAME


def test_logout_clears_session(api_client):
    api_client.post("/auth/login", json={"username": API_TEST_USERNAME, "password": API_TEST_PASSWORD})
    api_client.post("/auth/logout")
    resp = api_client.get("/auth/me")
    assert resp.status_code == 401
