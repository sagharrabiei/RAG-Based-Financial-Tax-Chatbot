import pytest
from fastapi.testclient import TestClient  #test client pretends to be a real browser calling my api
from api import app #importing my fast api app


client = TestClient(app)

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running!"}

def test_ask_answer():
    """tests that /ask endpoint return an answer"""
    response = client.post("/ask", json={"question": "مالیات بر ارزش افزوده جقدر است؟"})

    assert response.status_code == 200
    assert "answer" in response.json()
    assert len(response.json()["answer"]) > 0 #answer is not empty

def test_ask_empty_question():
    """test that if the user send an empty question the api return a response without crashing"""

    response = client.post("/ask", json={"question":""})
    assert response.status_code in [200,422]

def test_ask_wrong_input():
    """test if sending a wrong input returns a validation error"""
    resonse = client.post("/ask",json={"wrong_field":"this is wrong"})
    assert resonse.status_code == 422 #unprocessable entity

def test_ask_answer_is_string():
    """test question is actually a string"""
    response = client.post("/ask",json={"question":"مالیات بر ارزش افزوده چقدر است؟"})

    assert response.status_code == 200
    assert isinstance(response.json()["answer"], str)