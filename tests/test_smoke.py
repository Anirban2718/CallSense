import unittest

import config
from app import app


class FlaskSmokeTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "version": "1.0.0"})

    def test_runtime_directories_exist(self):
        self.assertTrue(config.UPLOAD_DIR.exists())
        self.assertTrue(config.RESULTS_DIR.exists())
        self.assertTrue(config.MODELS_DIR.exists())


if __name__ == "__main__":
    unittest.main()
