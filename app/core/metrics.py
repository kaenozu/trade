"""Advanced metrics collection and monitoring."""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .config import settings


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: Union[int, float], tags: Dict[str, str] = None):
        """Add a data point to the series."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get the latest data point."""
        return self.points[-1] if self.points else None
    
    def get_average(self, window_minutes: int = 5) -> Optional[float]:
        """Get average value over a time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        relevant_points = [p for p in self.points if p.timestamp >= cutoff]
        
        if not relevant_points:
            return None
        
        return sum(p.value for p in relevant_points) / len(relevant_points)
    
    def get_count(self, window_minutes: int = 5) -> int:
        """Get count of data points in a time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        return sum(1 for p in self.points if p.timestamp >= cutoff)


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self):
        self.series: Dict[str, MetricSeries] = {}
        self.prometheus_registry = None
        self.prometheus_metrics = {}
        
        if PROMETHEUS_AVAILABLE and settings.metrics_enabled:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics."""
        self.prometheus_registry = CollectorRegistry()
        
        # Core application metrics
        self.prometheus_metrics.update({
            'http_requests_total': Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status_code'],
                registry=self.prometheus_registry
            ),
            'http_request_duration_seconds': Histogram(
                'http_request_duration_seconds',
                'HTTP request duration in seconds',
                ['method', 'endpoint'],
                registry=self.prometheus_registry
            ),
            'data_fetch_duration_seconds': Histogram(
                'data_fetch_duration_seconds',
                'Data fetch duration in seconds',
                ['ticker', 'data_type'],
                registry=self.prometheus_registry
            ),
            'model_prediction_duration_seconds': Histogram(
                'model_prediction_duration_seconds',
                'Model prediction duration in seconds',
                ['ticker'],
                registry=self.prometheus_registry
            ),
            'cache_operations_total': Counter(
                'cache_operations_total',
                'Total cache operations',
                ['operation', 'hit_miss'],
                registry=self.prometheus_registry
            ),
            'active_connections': Gauge(
                'active_connections',
                'Number of active connections',
                registry=self.prometheus_registry
            ),
            'memory_usage_bytes': Gauge(
                'memory_usage_bytes',
                'Memory usage in bytes',
                ['type'],
                registry=self.prometheus_registry
            ),
            'prediction_accuracy': Gauge(
                'prediction_accuracy',
                'Model prediction accuracy',
                ['ticker', 'metric_type'],
                registry=self.prometheus_registry
            )
        })
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        tags: Dict[str, str] = None
    ):
        """Record a generic metric."""
        if name not in self.series:
            self.series[name] = MetricSeries(name)
        
        self.series[name].add_point(value, tags)
    
    def increment_counter(self, name: str, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, 1, tags)
    
    def record_timing(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record_metric(f"{name}_duration", duration_seconds, tags)
        
        # Also record to Prometheus if available
        if self.prometheus_registry and name in self.prometheus_metrics:
            # Prometheus metrics use labels, not keyword arguments
            metric = self.prometheus_metrics[name]
            if tags:
                metric.labels(**tags).observe(duration_seconds)
            else:
                metric.observe(duration_seconds)
    
    def record_http_request(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int,
        duration_seconds: float
    ):
        """Record HTTP request metrics."""
        tags = {
            'method': method.upper(),
            'endpoint': endpoint,
            'status_code': str(status_code)
        }
        
        self.increment_counter('http_requests_total', tags)
        self.record_timing('http_request_duration_seconds', duration_seconds, {
            'method': method.upper(),
            'endpoint': endpoint
        })
        
        # Record to Prometheus
        if self.prometheus_registry:
            self.prometheus_metrics['http_requests_total'].labels(
                method=method.upper(),
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.prometheus_metrics['http_request_duration_seconds'].labels(
                method=method.upper(),
                endpoint=endpoint
            ).observe(duration_seconds)
    
    def record_data_fetch(self, ticker: str, data_type: str, duration_seconds: float):
        """Record data fetch metrics."""
        tags = {'ticker': ticker, 'data_type': data_type}
        self.record_timing('data_fetch_duration_seconds', duration_seconds, tags)
        
        if self.prometheus_registry:
            self.prometheus_metrics['data_fetch_duration_seconds'].labels(
                ticker=ticker,
                data_type=data_type
            ).observe(duration_seconds)
    
    def record_model_prediction(self, ticker: str, duration_seconds: float):
        """Record model prediction metrics."""
        tags = {'ticker': ticker}
        self.record_timing('model_prediction_duration_seconds', duration_seconds, tags)
        
        if self.prometheus_registry:
            self.prometheus_metrics['model_prediction_duration_seconds'].labels(
                ticker=ticker
            ).observe(duration_seconds)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics."""
        tags = {
            'operation': operation,
            'hit_miss': 'hit' if hit else 'miss'
        }
        self.increment_counter('cache_operations_total', tags)
        
        if self.prometheus_registry:
            self.prometheus_metrics['cache_operations_total'].labels(
                operation=operation,
                hit_miss='hit' if hit else 'miss'
            ).inc()
    
    def update_system_metrics(self):
        """Update system-level metrics."""
        try:
            import psutil
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.record_metric('memory_rss_bytes', memory_info.rss)
            self.record_metric('memory_vms_bytes', memory_info.vms)
            
            if self.prometheus_registry:
                self.prometheus_metrics['memory_usage_bytes'].labels(type='rss').set(memory_info.rss)
                self.prometheus_metrics['memory_usage_bytes'].labels(type='vms').set(memory_info.vms)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.record_metric('cpu_percent', cpu_percent)
            
        except ImportError:
            pass  # psutil not available
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all metrics."""
        summary = {}
        
        for name, series in self.series.items():
            latest = series.get_latest()
            avg_5min = series.get_average(5)
            count_5min = series.get_count(5)
            
            summary[name] = {
                'latest_value': latest.value if latest else None,
                'latest_timestamp': latest.timestamp.isoformat() if latest else None,
                'avg_5min': avg_5min,
                'count_5min': count_5min,
                'total_points': len(series.points)
            }
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        if not self.prometheus_registry:
            return ""
        
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager to time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(operation_name, duration, tags)


class MetricsMiddleware:
    """Middleware to automatically collect HTTP metrics."""
    
    def __init__(self, app):
        self.app = app
        self.metrics = get_metrics_collector()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        status_code = 500  # Default to error
        
        async def send_with_metrics(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)
        
        try:
            await self.app(scope, receive, send_with_metrics)
        finally:
            duration = time.time() - start_time
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "/unknown")
            
            self.metrics.record_http_request(method, path, status_code, duration)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def setup_metrics_middleware(app):
    """Setup metrics collection middleware."""
    if settings.metrics_enabled:
        app.add_middleware(MetricsMiddleware)


# Convenience functions
def record_metric(name: str, value: Union[int, float], tags: Dict[str, str] = None):
    """Record a metric using the global collector."""
    get_metrics_collector().record_metric(name, value, tags)


def time_operation(operation_name: str, tags: Dict[str, str] = None):
    """Time an operation using the global collector."""
    return get_metrics_collector().time_operation(operation_name, tags)