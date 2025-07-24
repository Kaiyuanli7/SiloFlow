"""
Performance monitoring and metrics collection for SiloFlow pipeline service.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    active_requests: int
    total_requests: int
    avg_response_time: float
    cache_hit_ratio: float
    error_rate: float
    throughput_req_per_sec: float


class PerformanceMonitor:
    """
    Real-time performance monitoring for the SiloFlow pipeline service.
    
    Features:
    - System resource monitoring (CPU, memory, disk)
    - Request-level performance tracking
    - Cache performance analysis
    - Automatic performance alerts
    - Metrics export for analysis
    """
    
    def __init__(self, max_history: int = 1000, alert_thresholds: Optional[Dict] = None):
        self.max_history = max_history
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 85.0,
            'memory_percent': 80.0,
            'error_rate': 5.0,
            'avg_response_time': 10.0
        }
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.request_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.active_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.last_alert_time = {}
        
        logger.info("PerformanceMonitor initialized")
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info(f"Starting performance monitoring (interval: {interval}s)")
        
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    async def _collect_metrics(self):
        """Collect current performance metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Request metrics
            avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0.0
            error_rate = (self.error_count / max(1, self.total_requests)) * 100
            cache_hit_ratio = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            
            # Throughput calculation (requests in last minute)
            current_time = time.time()
            recent_requests = [t for t in self.request_times if current_time - t < 60]
            throughput = len(recent_requests) / 60.0
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                active_requests=self.active_requests,
                total_requests=self.total_requests,
                avg_response_time=avg_response_time,
                cache_hit_ratio=cache_hit_ratio,
                error_rate=error_rate,
                throughput_req_per_sec=throughput
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check for alerts
            await self._check_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds."""
        current_time = time.time()
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            if self._should_send_alert('cpu', current_time):
                logger.warning(f"[ALERT] HIGH CPU USAGE: {metrics.cpu_percent:.1f}%")
                self.last_alert_time['cpu'] = current_time
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            if self._should_send_alert('memory', current_time):
                logger.warning(f"[ALERT] HIGH MEMORY USAGE: {metrics.memory_percent:.1f}%")
                self.last_alert_time['memory'] = current_time
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            if self._should_send_alert('error_rate', current_time):
                logger.warning(f"[ALERT] HIGH ERROR RATE: {metrics.error_rate:.1f}%")
                self.last_alert_time['error_rate'] = current_time
        
        # Response time alert
        if metrics.avg_response_time > self.alert_thresholds['avg_response_time']:
            if self._should_send_alert('response_time', current_time):
                logger.warning(f"[ALERT] SLOW RESPONSE TIME: {metrics.avg_response_time:.2f}s")
                self.last_alert_time['response_time'] = current_time
    
    def _should_send_alert(self, alert_type: str, current_time: float, cooldown: float = 300.0) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_time = self.last_alert_time.get(alert_type, 0)
        return current_time - last_time > cooldown
    
    def record_request_start(self, request_id: str):
        """Record the start of a request."""
        self.active_requests += 1
        self.total_requests += 1
        self.timers[request_id] = time.time()
    
    def record_request_end(self, request_id: str, success: bool = True):
        """Record the end of a request."""
        self.active_requests = max(0, self.active_requests - 1)
        
        if request_id in self.timers:
            response_time = time.time() - self.timers[request_id]
            self.request_times.append(response_time)
            del self.timers[request_id]
        
        if not success:
            self.error_count += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter."""
        self.counters[name] += value
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        return {
            "timestamp": latest.timestamp,
            "system": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_available_gb": latest.memory_available_gb,
            },
            "requests": {
                "active": latest.active_requests,
                "total": latest.total_requests,
                "avg_response_time": latest.avg_response_time,
                "error_rate": latest.error_rate,
                "throughput_req_per_sec": latest.throughput_req_per_sec,
            },
            "cache": {
                "hit_ratio": latest.cache_hit_ratio,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
            "counters": dict(self.counters),
            "alerts": {
                "thresholds": self.alert_thresholds,
                "last_alerts": self.last_alert_time,
            }
        }
    
    def get_performance_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Filter metrics to window
        window_metrics = [m for m in self.metrics_history if m.timestamp >= window_start]
        
        if not window_metrics:
            return {"status": "no_data_in_window"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in window_metrics]
        memory_values = [m.memory_percent for m in window_metrics]
        response_times = [m.avg_response_time for m in window_metrics]
        
        return {
            "window_minutes": window_minutes,
            "samples": len(window_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "max": max(response_times),
                "min": min(response_times),
            },
            "requests": {
                "total": window_metrics[-1].total_requests - window_metrics[0].total_requests,
                "avg_throughput": sum(m.throughput_req_per_sec for m in window_metrics) / len(window_metrics),
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        try:
            metrics_data = []
            for metrics in self.metrics_history:
                metrics_data.append({
                    "timestamp": metrics.timestamp,
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "memory_available_gb": metrics.memory_available_gb,
                    "active_requests": metrics.active_requests,
                    "total_requests": metrics.total_requests,
                    "avg_response_time": metrics.avg_response_time,
                    "cache_hit_ratio": metrics.cache_hit_ratio,
                    "error_rate": metrics.error_rate,
                    "throughput_req_per_sec": metrics.throughput_req_per_sec,
                })
            
            export_data = {
                "export_time": time.time(),
                "metrics": metrics_data,
                "counters": dict(self.counters),
                "summary": self.get_performance_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


# Decorator for automatic request tracking
def track_performance(func):
    """Decorator to automatically track function performance."""
    async def wrapper(*args, **kwargs):
        request_id = f"{func.__name__}_{int(time.time() * 1000)}"
        monitor = get_performance_monitor()
        
        monitor.record_request_start(request_id)
        try:
            result = await func(*args, **kwargs)
            monitor.record_request_end(request_id, success=True)
            return result
        except Exception as e:
            monitor.record_request_end(request_id, success=False)
            raise
    
    return wrapper
