"""Health check and system information endpoints."""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from ..core.config import settings
from ..core.health import get_health_service
from ..core.metrics import get_metrics_collector
from ..models.api_models import HealthResponse, VersionResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system components."""
    health_service = get_health_service()
    return await health_service.get_overall_health()


@router.get("/health/live")
def liveness_probe():
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes-style readiness probe."""
    health_service = get_health_service()
    overall_health = await health_service.get_overall_health()
    
    # Ready if not unhealthy
    is_ready = overall_health["overall_status"] != "unhealthy"
    
    if is_ready:
        return {"status": "ready"}
    else:
        return Response(
            content='{"status": "not ready"}',
            status_code=503,
            media_type="application/json"
        )


@router.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Get application version information."""
    return VersionResponse(
        app=settings.app_name,
        version=settings.app_version,
        git_sha=settings.git_sha
    )


@router.get("/metrics-summary")
def get_metrics_summary():
    """Get application metrics summary."""
    metrics = get_metrics_collector()
    metrics.update_system_metrics()  # Update current system metrics
    return metrics.get_metrics_summary()


@router.get("/metrics", response_class=PlainTextResponse)
def get_prometheus_metrics():
    """Get Prometheus formatted metrics."""
    metrics = get_metrics_collector()
    metrics.update_system_metrics()  # Update current system metrics
    return metrics.get_prometheus_metrics()