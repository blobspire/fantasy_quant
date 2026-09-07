"""ESPN Fantasy endpoint constants.

Host choice is not cosmetic. `fantasy.espn.com/apis/v3/...` has 302'd to a marketing
page since April 2024, and `site.api.espn.com` returns 403 from datacenter IPs
regardless of headers. `lm-api-reads` is the one that answers.
"""

from __future__ import annotations

READ_HOST = "https://lm-api-reads.fantasy.espn.com"
BASE = f"{READ_HOST}/apis/v3/games/ffl"
FAN_API = "https://fan.api.espn.com/apis/v2/fans"

# A browser UA is genuinely sufficient for lm-api-reads; no token dance is needed.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Public, league-independent player pools already scored in a known context.
# No auth, no league id, no key.
LEAGUE_DEFAULTS = {
    "standard": 1,
    "ppr": 3,
    "half_ppr": 8,
}


def league_default_url(season: int, variant: str = "ppr") -> str:
    """Player pool scored under one of ESPN's canned scoring settings."""
    if variant not in LEAGUE_DEFAULTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(LEAGUE_DEFAULTS)}")
    return f"{BASE}/seasons/{season}/segments/0/leaguedefaults/{LEAGUE_DEFAULTS[variant]}"


def league_url(season: int, league_id: int) -> str:
    return f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"


def platform_settings_url(season: int) -> str:
    """ESPN's own machine-readable constants dictionary (235 statIds, slots, enums)."""
    return f"{BASE}/seasons/{season}?view=chui_default_platformsettings"


def pro_schedule_url(season: int) -> str:
    return f"{BASE}/seasons/{season}?view=proTeamSchedules_wl"


def game_meta_url() -> str:
    """currentSeasonId + currentScoringPeriod. Cheap; use it instead of computing the week."""
    return BASE
