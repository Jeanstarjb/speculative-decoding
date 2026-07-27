import redis
import time

class MonitoringSystem:
    """Real-time performance monitoring system for speculative decoding"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_conn = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        self._init_metrics()

    def _init_metrics(self):
        metrics = [
            'total_tokens', 'total_processing_time',
            'accepted_tokens', 'speculated_tokens',
            'latency_sum', 'latency_count'
        ]
        for metric in metrics:
            if not self.redis_conn.exists(f'metrics:{metric}'):
                self.redis_conn.set(f'metrics:{metric}', 0)

    def increment_accepted_tokens(self, count=1):
        self.redis_conn.incrby('metrics:accepted_tokens', count)

    def increment_speculated_tokens(self, count=1):
        self.redis_conn.incrby('metrics:speculated_tokens', count)

    def record_latency(self, latency):
        self.redis_conn.incrbyfloat('metrics:latency_sum', latency)
        self.redis_conn.incr('metrics:latency_count', 1)

    def record_request_metrics(self, num_tokens, duration):
        with self.redis_conn.pipeline() as pipe:
            pipe.incrby('metrics:total_tokens', num_tokens)
            pipe.incrbyfloat('metrics:total_processing_time', duration)
            pipe.execute()

    def get_metrics(self):
        metrics = {
            'total_tokens': int(self.redis_conn.get('metrics:total_tokens') or 0),
            'total_processing_time': float(self.redis_conn.get('metrics:total_processing_time') or 0),
            'accepted_tokens': int(self.redis_conn.get('metrics:accepted_tokens') or 0),
            'speculated_tokens': int(self.redis_conn.get('metrics:speculated_tokens') or 0),
            'latency_sum': float(self.redis_conn.get('metrics:latency_sum') or 0),
            'latency_count': int(self.redis_conn.get('metrics:latency_count') or 0),
        }

        throughput = metrics['total_tokens'] / metrics['total_processing_time'] if metrics['total_processing_time'] > 0 else 0
        avg_latency = metrics['latency_sum'] / metrics['latency_count'] if metrics['latency_count'] > 0 else 0
        acceptance_rate = metrics['accepted_tokens'] / metrics['speculated_tokens'] if metrics['speculated_tokens'] > 0 else 0

        return {
            'throughput_tokens_per_sec': throughput,
            'average_latency_per_step_sec': avg_latency,
            'acceptance_rate': acceptance_rate
        }