import pytest
from unittest.mock import Mock
from backend.app.monitoring import MonitoringSystem

@pytest.fixture
def mock_redis():
    redis_mock = Mock()
    redis_mock.get.side_effect = lambda key: 0
    redis_mock.incrby = Mock()
    redis_mock.incrbyfloat = Mock()
    redis_mock.incr = Mock()
    redis_mock.exists.return_value = False
    redis_mock.set = Mock()
    return redis_mock

def test_initialization(mock_redis):
    MonitoringSystem(redis_conn=mock_redis)
    assert mock_redis.set.call_count == 6

def test_metrics_recording(mock_redis):
    monitor = MonitoringSystem(redis_conn=mock_redis)
    
    monitor.increment_accepted_tokens(5)
    mock_redis.incrby.assert_called_with('metrics:accepted_tokens', 5)
    
    monitor.increment_speculated_tokens(3)
    mock_redis.incrby.assert_called_with('metrics:speculated_tokens', 3)
    
    monitor.record_latency(0.15)
    mock_redis.incrbyfloat.assert_called_with('metrics:latency_sum', 0.15)
    mock_redis.incr.assert_called_with('metrics:latency_count', 1)
    
    monitor.record_request_metrics(10, 2.5)
    mock_redis.incrby.assert_called_with('metrics:total_tokens', 10)
    mock_redis.incrbyfloat.assert_called_with('metrics:total_processing_time', 2.5)

def test_metrics_calculation(mock_redis):
    mock_redis.get.side_effect = lambda key: {
        'metrics:total_tokens': b'100',
        'metrics:total_processing_time': b'20.0',
        'metrics:accepted_tokens': b'75',
        'metrics:speculated_tokens': b'100',
        'metrics:latency_sum': b'15.0',
        'metrics:latency_count': b'50'
    }.get(key, b'0')
    
    monitor = MonitoringSystem(redis_conn=mock_redis)
    metrics = monitor.get_metrics()
    
    assert metrics['throughput_tokens_per_sec'] == 5.0
    assert metrics['average_latency_per_step_sec'] == 0.3
    assert metrics['acceptance_rate'] == 0.75