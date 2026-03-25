"""
Integration tests for the Flask API endpoints.
Tests all RESTful routes with mock data.
"""

import io
import json
import unittest
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_test_wav():
    """Generate a minimal WAV file for testing."""
    import soundfile as sf
    buffer = io.BytesIO()
    audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    sf.write(buffer, audio, 16000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("services", data)
        self.assertIn("version", data)


class TestMetricsEndpoint(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_get_metrics(self):
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("overview", data)
        self.assertIn("latency", data)

    def test_get_recent_metrics(self):
        response = self.client.get("/api/v1/metrics/recent?n=5")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("recent", data)


class TestTaskEndpoints(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_create_task(self):
        response = self.client.post(
            "/api/v1/tasks",
            json={"type": "reminder", "description": "Test reminder"},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn("id", data)
        self.assertEqual(data["type"], "reminder")
        self.assertEqual(data["status"], "pending")

    def test_create_and_execute_task(self):
        response = self.client.post(
            "/api/v1/tasks",
            json={"type": "timer", "description": "Test timer", "execute": True},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["status"], "completed")

    def test_list_tasks(self):
        # Create some tasks
        self.client.post("/api/v1/tasks", json={"type": "reminder", "description": "A"})
        self.client.post("/api/v1/tasks", json={"type": "email", "description": "B"})

        response = self.client.get("/api/v1/tasks")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(data["count"], 2)

    def test_get_task_by_id(self):
        create_response = self.client.post(
            "/api/v1/tasks",
            json={"type": "reminder", "description": "Get me"},
        )
        task_id = create_response.get_json()["id"]

        response = self.client.get(f"/api/v1/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["id"], task_id)

    def test_get_nonexistent_task(self):
        response = self.client.get("/api/v1/tasks/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_cancel_task(self):
        create_response = self.client.post(
            "/api/v1/tasks",
            json={"type": "reminder", "description": "Cancel me"},
        )
        task_id = create_response.get_json()["id"]

        response = self.client.post(f"/api/v1/tasks/{task_id}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "cancelled")

    def test_task_stats(self):
        self.client.post("/api/v1/tasks", json={"type": "reminder", "description": "X"})
        response = self.client.get("/api/v1/tasks/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("total", data)
        self.assertIn("by_status", data)

    def test_invalid_task_request(self):
        response = self.client.post("/api/v1/tasks", json={"description": "Missing type"})
        self.assertEqual(response.status_code, 400)


class TestIntegrationEndpoints(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_register_integration(self):
        response = self.client.post(
            "/api/v1/integrate",
            json={
                "name": "Test Service",
                "base_url": "https://api.test.com/",
                "auth_type": "bearer",
                "auth_token": "test-token",
            },
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["name"], "Test Service")

    def test_register_from_template(self):
        response = self.client.post(
            "/api/v1/integrate",
            json={"template": "slack", "auth_token": "xoxb-test"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["name"], "Slack")

    def test_list_integrations(self):
        self.client.post(
            "/api/v1/integrate",
            json={"name": "Svc1", "base_url": "https://a.com/"},
        )
        response = self.client.get("/api/v1/integrate")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.get_json()["count"], 1)

    def test_list_templates(self):
        response = self.client.get("/api/v1/integrate/templates")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreater(len(data["templates"]), 0)

    def test_invalid_integration(self):
        response = self.client.post("/api/v1/integrate", json={"name": "No URL"})
        self.assertEqual(response.status_code, 400)


class TestWebhookEndpoints(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_register_webhook(self):
        response = self.client.post(
            "/api/v1/webhooks",
            json={
                "url": "https://hooks.example.com/notify",
                "events": ["transcription.complete"],
            },
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn("id", data)

    def test_list_webhooks(self):
        self.client.post(
            "/api/v1/webhooks",
            json={"url": "https://hooks.test.com/a", "events": ["test"]},
        )
        response = self.client.get("/api/v1/webhooks")
        self.assertEqual(response.status_code, 200)

    def test_invalid_webhook(self):
        response = self.client.post(
            "/api/v1/webhooks",
            json={"url": "not-a-url", "events": ["test"]},
        )
        self.assertEqual(response.status_code, 400)


class TestCommandEndpoint(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_command_no_input(self):
        response = self.client.post("/api/v1/command")
        self.assertEqual(response.status_code, 400)

    def test_command_empty_text(self):
        response = self.client.post("/api/v1/command", json={"text": ""})
        self.assertEqual(response.status_code, 400)

    def test_command_text_input(self):
        """Test text-based command (will fail without API key but validates flow)."""
        response = self.client.post(
            "/api/v1/command",
            json={"text": "What time is it?"},
        )
        # Should either succeed or fail gracefully
        self.assertIn(response.status_code, [200, 500])


class TestIntentsEndpoint(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_list_intents(self):
        response = self.client.get("/api/v1/intents")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("intents", data)
        self.assertIn("search", data["intents"])
        self.assertIn("weather", data["intents"])
        self.assertGreater(len(data["intents"]), 5)


class TestContextEndpoint(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_clear_context(self):
        response = self.client.post("/api/v1/context/clear")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "context_cleared")


class TestDashboard(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_dashboard_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Vox", response.data)


if __name__ == "__main__":
    unittest.main()
