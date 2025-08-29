"""Health check and system information endpoints."""

from fastapi import APIRouter

from ..core.config import settings
from ..models.api_models import HealthResponse, VersionResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Get application version information."""
    return VersionResponse(
        app=settings.app_name,
        version=settings.app_version,
        git_sha=settings.git_sha
    )