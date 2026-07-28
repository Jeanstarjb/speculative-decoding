from locust import HttpUser, task, between
import random

class StressTestUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def generate_text(self):
        prompts = [
            "Explain quantum computing in simple terms",
            "Write a poem about AI ethics",
            "Summarize the latest climate change findings",
            "Generate Python code for quick sort"
        ]
        self.client.post(
            "/generate",
            json={
                "prompt": random.choice(prompts),
                "max_length": 128,
                "temperature": 0.7
            }
        )

    @task(3)
    def health_check(self):
        self.client.get("/health")
