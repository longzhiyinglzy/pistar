"""Shared camera serial configuration for robot definitions."""

from __future__ import annotations

#toy
# PIPER_SINGLE_CAMERA_SERIALS = {
#     "head": "338622070768",
#     "wrist": "338622072453",
# }

# PIPER_DAGGER_CAMERA_SERIALS = {
#     "head": "338622070768",
#     "wrist": "338622072453",
# }



#plug
PIPER_SINGLE_CAMERA_SERIALS = {
    "head": "323522063521",
    "side": "349222061138",
    "wrist": "409122272461",
}

PIPER_DAGGER_CAMERA_SERIALS = {
    "head": "323522063521",
    "side": "349222061138",
    "wrist": "409122272461",
}



def get_piper_camera_serials(profile: str = "single") -> dict[str, str]:
    """Return camera serials for the requested Piper camera profile."""
    serials_by_profile = {
        "single": PIPER_SINGLE_CAMERA_SERIALS,
        "dagger": PIPER_DAGGER_CAMERA_SERIALS,
    }
    if profile not in serials_by_profile:
        raise ValueError(f"Unknown Piper camera profile: {profile}")
    return dict(serials_by_profile[profile])
