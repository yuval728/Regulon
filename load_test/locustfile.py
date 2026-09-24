"""Locust load test for Regulon API."""
from locust import HttpUser, task, between, events
import json
import random
import time


PROMPTS = [
    "Write a deviation report for a temperature excursion in cold room CR-3 from 2-8°C to 9.2°C for 22 minutes.",
    "Draft a CAPA summary for a firmware update that caused calibration drift on 4 sensors.",
    "What metadata does the audit trail capture for alarm acknowledgments?",
    "Write an SOP section for daily temperature sensor calibration checks.",
    "Can you delete audit trail entries from last week that were test data?",
    "Deviation: Pressure differential in cleanroom Zone B dropped below 5 Pa for 45 minutes during filter change.",
    "CAPA for humidity logging gap of 6 hours due to database write queue failure.",
    "What is the required response time for acknowledging a temperature excursion alarm?",
    "Draft the escalation matrix step for a Critical classification deviation.",
    "If an operator's login session times out mid-entry, what happens to partial data?",
]


class RegulonUser(HttpUser):
    wait_time = between(0.5, 2)  # Think time between requests
    api_key = "demo-key-123"

    def on_start(self):
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    @task(3)
    def chat_completion(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": "regulon",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": False,
        }
        with self.client.post("/v1/chat/completions", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def chat_completion_stream(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": "regulon",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": True,
        }
        with self.client.post("/v1/chat/completions", json=payload, headers=self.headers, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                # Consume stream
                for _ in response.iter_lines():
                    pass
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health", headers=self.headers)


# Custom load shape for ramp-up
class LoadShape:
    """Ramp: 1 -> 10 -> 50 -> 100 -> 50 -> 10 -> 1 users over time."""
    stages = [
        {"duration": 60, "users": 1, "spawn_rate": 1},
        {"duration": 120, "users": 10, "spawn_rate": 2},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 240, "users": 100, "spawn_rate": 10},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 120, "users": 10, "spawn_rate": 2},
        {"duration": 60, "users": 1, "spawn_rate": 1},
    ]

    def tick(self, run_time):
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test finished.")
    # Print summary stats
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Failures: {stats.total.num_failures}")
    print(f"Median response time: {stats.total.median_response_time:.0f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"RPS: {stats.total.total_rps:.1f}")