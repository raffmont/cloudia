"""WordPress publishing helpers for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for base64 encoding.
import base64
# Standard library import for logging.
import logging
# Standard library import for os environment variables.
import os
# Standard library import for typing helpers.
from typing import Any, Dict, Optional

# Local import for config model.
from cloudia_config import AppConfig
# Local import for request session helper.
from cloudia_utils import die, make_requests_session

# Module-level logger for WordPress helper.
logger = logging.getLogger(__name__)


def wp_auth_header(user: str, app_password: str) -> Dict[str, str]:
    """Build a basic auth header for WordPress."""
    # Encode username and password in base64.
    token = base64.b64encode(f"{user}:{app_password}".encode("utf-8")).decode("ascii")
    # Return the Authorization header.
    return {"Authorization": f"Basic {token}"}


def wp_create_post(
    cfg: AppConfig,
    title: str,
    md: str,
    status: str = "draft",
    featured_image: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a WordPress post with provided content."""
    # Enforce that publishing is enabled.
    if not cfg.wp.enabled:
        # Provide a clear error if publishing is disabled.
        die("WordPress publishing is disabled in config (wordpress.enabled=false).")
    # Enforce required base URL configuration.
    if not cfg.wp.base_url:
        # Provide a clear error if base URL is missing.
        die("wordpress.base_url is required to publish.")
    # Resolve WordPress credentials from config or environment.
    user = cfg.wp.user or os.getenv("WP_USER")
    # Resolve the WordPress app password from config or environment.
    pwd = cfg.wp.app_password or os.getenv("WP_APP_PASSWORD")
    # Validate credentials are present.
    if not user or not pwd:
        # Provide a clear error for missing credentials.
        die("WordPress credentials missing (WP_USER/WP_APP_PASSWORD or config wordpress.user/app_password).")

    # Build the posts endpoint URL.
    url = cfg.wp.base_url.rstrip("/") + "/posts"
    # Construct the JSON payload for the post.
    payload: Dict[str, Any] = {
        "title": title,
        "content": md,
        "status": status,
        "categories": cfg.wp.category_ids or [25],
    }
    # Add a featured image when provided.
    if featured_image is not None:
        # Ensure the featured image is an integer id.
        payload["featured_media"] = int(featured_image)

    # Create a requests session with configured timeout.
    session = make_requests_session(cfg.api.http_timeout_sec)
    # Perform the post creation request.
    response = session.post(url, json=payload, headers=wp_auth_header(user, pwd))
    # Handle error responses with clear context.
    if response.status_code >= 300:
        # Surface the error status and a preview of the response.
        die(f"WordPress publish failed ({response.status_code}): {response.text[:500]}")
    # Return the JSON response for downstream usage.
    return response.json()
