"""The surface the user actually touches: one report per league, one queue across them.

Everything below this file answers one question well and returns a `core.Recommendation`.
This module answers the only question the user ever asks -- *what should I do this week?*
-- by running those surfaces against a live league and putting their answers in one
order. It computes nothing of its own beyond arithmetic on numbers the surfaces already
published, and it is the only place that knows all of them exist.

Four rules shape it, and each one is a decision that could have gone the other way.

**The terminal and `--json` read the same dictionary.** Every command builds a payload
first and renders *from that payload*; nothing is computed inside a `rich` table. The
dashboard consuming `--json` therefore sees exactly the numbers the terminal shows, and
the two cannot drift, because there is only one of them. This is why the render
functions take a `dict` rather than a `LeagueSim`.

**Each surface's own significance test wins, and none of them is `Recommendation.significant`.**
`core.Recommendation.significant` is a two-sigma test against `stderr`, and it reads
`stderr == 0` as certainty -- which is right for an analytic result and wrong for the
three cases that actually produce it here: a claim worth exactly nothing, a lineup
already set, and a streaming plan identical to holding. Worse, two sigma is the wrong
test for the candidate that *won a search*: `decide/trades.py` measured its top trade at
+1.12pp +/- 0.49 (2.3 sigma, "significant") where forty thousand simulations put it at
+0.53pp +/- 0.13. So this module asks each surface its own question --
`waivers.ClaimPrice.significant`, `lineups.LineupAdvice.significant`, and for trades the
selection-adjusted `confidence != "low"` -- and `verdict_for` is the one place that
knows which is which. A recommendation inside its own error is rendered dim with a `~`
and never silently dropped.

**A hold is an answer, and it is written out in numbers.** "Nothing on the wire beats
your worst starter and this week is a coin flip, so a marginal upgrade is worth under
0.1pp" is a useful output. `headline` composes exactly that sentence from the measured
threshold, the measured leverage and the measured points-to-title rate rather than from a
template, so it cannot claim a hold the numbers do not support.

**Levels across surfaces are still not comparable; deltas are.** They used to disagree
about the FLOOR as well: `pipeline.championship_table` scored an unfilled starting slot at
zero while `decide/waivers.py` streamed a replacement into it, so one league carried two
published baselines, disclosed only as a footer string. Every surface now floors an empty
seat at the wire and they agree about the convention.

What is left is not a bug and will not be fixed: each surface draws its own season.
`championship_table` runs on `pipeline.build`'s draw, `decide/waivers` on a widened panel
that gives free agents columns, `edges/portfolio` on one seed shared across three leagues.
Those are genuinely different Monte Carlo universes, so their LEVELS will never match to
the decimal, and only `delta_title` travels between leagues -- which is the entire reason
the queue can exist.

The commands (`odds`, `weekly`, `waivers`, `trades`, `lineup`, `stream`, `queue`) all
take `--league` by id or by registry name, repeatable, defaulting to every enabled league
in the registry, and all take `--json`. A league that cannot be fetched degrades to one
red row and an `error` string in its payload; it never takes the other leagues down.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import corpus, pipeline, registry
from .core import DST, Recommendation
from .data import etr
from .decide import lineups as lineups_mod
from .decide import streaming as streaming_mod
from .decide import title as title_mod
from .decide import trades as trades_mod
from .decide import waivers as waivers_mod
from .decide.valuation import POSITION_ABBREV
from .espn.endpoints import LEAGUE_DEFAULTS
from .projections.calibration import load as load_calibration

log = logging.getLogger(__name__)

#: Bumped when a `--json` payload changes shape in a way a consumer would notice. New
#: keys do not bump it; renamed or removed keys do.
SCHEMA_VERSION = 1

#: Sections `fq weekly` runs, in the order it prints them. `--skip` takes these names.
WEEKLY_SECTIONS: tuple[str, ...] = ("odds", "leverage", "lineup", "waivers", "trades", "stream")

#: The single hook `edges/portfolio.py` may implement to take over queue ordering:
#:
#:     def action_queue(actions: Sequence[Action], *, limit: int) -> Sequence[Action]
#:
#: Looked up by name at call time, so the module being written concurrently is optional
#: rather than a hard dependency. Anything unexpected there falls back to `rank_actions`
#: and says so in `ranked_by`, because a queue silently ordered by something other than
#: what it claims is worse than no queue.
PORTFOLIO_HOOK = "action_queue"

#: Below this the marginal point is worth under a quarter of its coin-flip value and the
#: week is effectively decided. `decide/title.WeekLeverage.decided` is the same constant;
#: it is repeated here only so the rendering can colour on it without importing the row.
DECIDED_LEVERAGE = 0.25

console = Console()


class ReportError(RuntimeError):
    """The report cannot be assembled as asked. Distinct from a league that failed."""


# --------------------------------------------------------------------------------------
# League selection
# --------------------------------------------------------------------------------------


def load_registry(path: Path | str | None = None) -> registry.Registry:
    """The league registry, or an empty one. `fq` is useless without it, so say so."""
    reg = registry.Registry.load(path or registry.DEFAULT_CONFIG_PATH)
    if not reg.leagues:
        raise ReportError(
            f"no leagues configured in {path or registry.DEFAULT_CONFIG_PATH}. "
            "Run `fq sync`, or write a config/leagues.toml with a [[leagues]] entry."
        )
    return reg


def select_leagues(
    reg: registry.Registry,
    selectors: Sequence[str] | None = None,
    *,
    season: int | None = None,
) -> list[registry.LeagueConfig]:
    """Resolve `--league` values against the registry. Ids and names both work.

    Names are matched case-insensitively, exact first and then as a prefix and then as a
    substring, so `--league wine` finds Wine Wednesday and `--league 272150391` finds it
    too. An ambiguous name raises rather than picking one: running the wrong league's
    waiver board is a mistake the user cannot see in the output.
    """
    pool = [c for c in reg.sorted() if season is None or c.season == season]
    if not selectors:
        chosen = [c for c in pool if c.enabled]
        if not chosen:
            raise ReportError(
                f"no enabled leagues in the registry for season {season or reg.defaults.season}"
            )
        return chosen

    known = ", ".join(f"{c.league_id} ({c.name})" for c in pool) or "nothing"
    out: list[registry.LeagueConfig] = []
    for raw in selectors:
        token = raw.strip()
        if token.isdigit():
            matches = [c for c in pool if c.league_id == int(token)]
        else:
            low = token.casefold()
            names = [(c, c.name.casefold()) for c in pool]
            matches = [c for c, n in names if n == low]
            matches = matches or [c for c, n in names if n.startswith(low)]
            matches = matches or [c for c, n in names if low in n]
        if not matches:
            raise ReportError(f"no league matches {token!r}; the registry holds {known}")
        if len(matches) > 1:
            hits = ", ".join(f"{c.league_id} ({c.name})" for c in matches)
            raise ReportError(f"{token!r} is ambiguous: {hits}. Use the league id.")
        if matches[0] not in out:
            out.append(matches[0])
    return out


# --------------------------------------------------------------------------------------
# The workspace
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Workspace:
    """One authenticated client and one built `LeagueSim` per league, reused.

    Deliberately not frozen and deliberately stateful: `pipeline.build` costs three
    seconds and a `TitleEngine` another half, and `fq weekly` needs both for every
    section. More importantly, **`pipeline.build` closes any client it opened itself**,
    which leaves `sim.league.rosters()` raising "Cannot send a request, as the client has
    been closed" -- so the currently-set lineup, the standings and the waiver order are
    all unreadable unless the caller owns the client. That is what this owns.

    Every league in one workspace shares a seed, so two runs of `fq weekly` a minute
    apart agree; two *different* workspaces are two different universes and their levels
    must not be differenced against each other.
    """

    season: int | None = None
    n_sims: int = 4000
    seed: int = 1
    _client: Any = None
    _sims: dict[tuple[int, int], Any] = field(default_factory=dict)
    _engines: dict[tuple[int, int], Any] = field(default_factory=dict)
    _teams: dict[tuple[int, int], Any] = field(default_factory=dict)
    _variants: dict[tuple[int, int], tuple[str, str]] = field(default_factory=dict)
    #: The analyst board this workspace prices the wire and the trade board against.
    #: `None` disables it; the default reads `data/manual/etr/`. See `rankings`.
    rankings_kind: str | None = "silva"
    rankings_dir: Path | str = etr.DEFAULT_DIR
    _rankings: dict[str, tuple[Any, bool]] = field(default_factory=dict)

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = pipeline.client_from_env()
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def variant(self, cfg: registry.LeagueConfig) -> tuple[str, str]:
        """`(the corpus variant that will be read, the one the registry asked for)`.

        Two of the user's three leagues declare `half_ppr` and the corpus holds only
        `ppr` for 2026 -- `leaguedefaults/8` is a 2026-only endpoint and the daily
        snapshot has not been capturing it -- so a strict reading loses two leagues out of
        three to a `CorpusError`.

        Substituting is not a fudge, and the reason is worth being precise about.
        `pipeline.league_projections` reads the raw **component stat lines** out of the
        corpus and applies THIS LEAGUE'S OWN scoring function to them; ESPN's canned
        variant only decides the `appliedTotal` column, which that function never reads.
        So a half-PPR league scored off the PPR capture gets exactly the half-PPR
        projections it would have got off a half-PPR capture. What genuinely differs is
        the *pool* -- ESPN can return a slightly different player list per canned setting
        -- and the calibration, which is why `sim` passes the league's true variant to
        `load_calibration` while reading the corpus from whichever variant exists.

        The substitution is reported in every payload rather than hidden; a report that
        quietly reads a different corpus than it says is a report that cannot be checked.
        """
        cached = self._variants.get(cfg.key)
        if cached is not None:
            return cached
        wanted = cfg.scoring_variant
        resolved = (wanted, wanted)
        if cfg.season not in corpus.available_seasons(variant=wanted):
            for candidate in LEAGUE_DEFAULTS:
                if candidate == wanted:
                    continue
                if cfg.season in corpus.available_seasons(variant=candidate):
                    log.warning(
                        "no %r corpus for season %d; reading %r and re-scoring for league %d",
                        wanted,
                        cfg.season,
                        candidate,
                        cfg.league_id,
                    )
                    resolved = (candidate, wanted)
                    break
        self._variants[cfg.key] = resolved
        return resolved

    def sim(self, cfg: registry.LeagueConfig) -> Any:
        """This league's `pipeline.LeagueSim`, built once per workspace."""
        cached = self._sims.get(cfg.key)
        if cached is None:
            read, wanted = self.variant(cfg)
            cached = pipeline.build(
                cfg.league_id,
                cfg.season,
                my_team_id=cfg.team_id,
                client=self.client,
                n_sims=self.n_sims,
                seed=self.seed,
                variant=read,
                # The league's own variant, not the one the corpus happened to have.
                # Only the stat rows are being substituted; the calibration is not.
                calibration=load_calibration(wanted),
                objective=cfg.objective.value,
            )
            self._sims[cfg.key] = cached
        return cached

    def engine(self, cfg: registry.LeagueConfig) -> Any:
        """A `decide/title.TitleEngine` on the same draw. Only `week_leverage` needs it.

        The surrogate fit is lazy inside the engine, so building one to read the leverage
        schedule costs the baseline season (~0.5s) and no surface fits.
        """
        cached = self._engines.get(cfg.key)
        if cached is None:
            cached = title_mod.TitleEngine.from_sim(self.sim(cfg))
            self._engines[cfg.key] = cached
        return cached

    def teams(self, cfg: registry.LeagueConfig) -> Mapping[int, Any]:
        """ESPN's own standings rows, keyed by team id. Empty when the fetch failed.

        The simulated table is the answer; this carries the facts the simulation has
        already folded in (record, points for, waiver order) and which the user checks
        the report against.
        """
        cached = self._teams.get(cfg.key)
        if cached is None:
            try:
                cached = {t.id: t for t in self.sim(cfg).league.teams().teams}
            except Exception as err:  # pragma: no cover - live-only path
                log.info("no standings for league %s (%s)", cfg.league_id, err)
                cached = {}
            self._teams[cfg.key] = cached
        return cached

    def rankings(self, cfg: registry.LeagueConfig) -> tuple[Any, bool]:
        """`(analyst board, matches this league's scoring)`, or `(None, False)`.

        Read once per scoring format and archived once per day on first use, because
        Establish The Run overwrites each chart in place and there is no other record of
        what the board said the week a decision was made against it.

        The half-PPR board is handed to the full-PPR league deliberately, and the flag
        says so. It is safe for these two consumers and would not be for a third: both
        read only `positional()`, and within a position the two formats' orderings agree
        to a Spearman of +0.9967 or better (measured on the two 300-row Draft Kit
        boards, `data/etr.py`). A consumer of the OVERALL rank must not take the
        fallback, because that is where half PPR and full PPR actually disagree.
        """
        if self.rankings_kind is None:
            return None, False
        cached = self._rankings.get(cfg.scoring_variant)
        if cached is not None:
            return cached
        board, matched = etr.best_available(
            cfg.scoring_variant, self.rankings_dir, kind=self.rankings_kind
        )
        if board is not None:
            if not matched:
                log.info(
                    "no %s %s board; using the %s one for league %d (within-position "
                    "orderings agree to rho >= 0.9967)",
                    self.rankings_kind,
                    cfg.scoring_variant,
                    board.scoring,
                    cfg.league_id,
                )
            try:
                etr.archive(board.path, root=Path(self.rankings_dir) / "archive")
            except OSError as err:  # pragma: no cover - disk-only path
                log.warning("could not archive %s: %s", board.path.name, err)
        self._rankings[cfg.scoring_variant] = (board, matched)
        return board, matched

    def my_team_id(self, cfg: registry.LeagueConfig) -> int:
        team_id = cfg.team_id if cfg.team_id is not None else self.sim(cfg).state.my_team_id
        if team_id is None:
            raise ReportError(
                f"league {cfg.league_id} has no team_id in the registry and ESPN did not "
                "identify one; every surface here advises a specific franchise."
            )
        return int(team_id)

    def names(self, cfg: registry.LeagueConfig) -> dict[int, str]:
        """player id -> name over the whole projected pool, free agents included.

        Off `sim.outlooks` rather than `state.pool`, because every surface that can name
        somebody the user does not roster -- a waiver add, a streaming candidate -- draws
        that player from the outlooks.
        """
        sim = self.sim(cfg)
        out = {o.player_id: o.name for o in sim.outlooks if o.name}
        for pid in sim.state.pool.player_ids:
            out.setdefault(int(pid), sim.state.pool.name(int(pid)))
        return out

    def current_starters(self, cfg: registry.LeagueConfig) -> tuple[int, ...] | None:
        """What the manager actually has in his lineup right now, or None if unreadable.

        `None` is not the same as "the projected-best lineup". A report that silently
        substituted the optimum would show "no change" to a manager who has not logged in
        since the draft, which is the one user this surface exists for.
        """
        try:
            rosters = self.sim(cfg).league.rosters()
        except Exception as err:  # pragma: no cover - live-only path
            log.info("no live roster for league %s (%s)", cfg.league_id, err)
            return None
        roster = rosters.get(self.my_team_id(cfg))
        if roster is None:
            return None
        return lineups_mod.starters_from_roster(roster)


# --------------------------------------------------------------------------------------
# Verdicts: which surface's significance test applies
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether an effect is real, noise, or nothing at all.

    Four states rather than a boolean because they need different words. `null` is a
    move that changes nothing -- a hold, an already-set lineup, a plan identical to the
    baseline -- and its zero standard error means *nothing was measured*, not *measured
    precisely*. `noise` is a real measurement that does not clear its own error, and it
    must be shown, because a user who is told nothing was found will go looking on his
    own. `harm` is a measurement that cleared its error *in the wrong direction*, which
    is not the same fact as `noise` and must not be printed as one: `decide/trades.py`
    returns `confidence="low"` both for a search winner inside its error and for a trade
    the simulation says costs the user title probability, and calling the second one
    "inside its own error" tells the reader the opposite of what was measured. `act` is
    the only one the queue ranks on.
    """

    kind: str
    note: str

    @property
    def significant(self) -> bool:
        return self.kind == "act"

    @property
    def marker(self) -> str:
        return {"act": "", "noise": "~", "null": "-", "harm": "!"}[self.kind]


#: Tags that mean *nothing was measured*, which is not the same as *nothing to do*.
#: `null-plan` is `decide/streaming.py`'s marker for a plan bit-identical to holding, so
#: its zero delta and zero error are structural. `no-action-this-week` is deliberately
#: NOT here: it means this week's move is a hold while the plan behind it is real and
#: measured, and treating a +2.77pp +/- 0.35 plan as unmeasured because today is quiet
#: throws away the only significant number in two of the three live leagues.
_NULL_TAGS = frozenset({"null-plan"})

#: Tags a surface attaches when it sets `confidence="low"`, named in the verdict note so
#: "the surface does not stand behind this" is a reason rather than an assertion.
_LOW_CONFIDENCE_TAGS = frozenset(
    {
        "not-streamable",
        "action-suppressed",
        "harmful",
        "harmful-unclear",
        "counterparty-loses",
        "counterparty-loses-unclear",
    }
)

#: Tags meaning the number is real and somebody else still has to agree to it, or that
#: part of it rests on an input that has not arrived. Surfaced as `caveats` on every
#: `Action` rather than left in `tags` for the reader to decode: `decide/trades.py`
#: writes `counterparty-loses` when the simulation says a counterparty's title odds FALL
#: under a trade its own points gate called Pareto-improving, and its rationale calls
#: printing that without saying so "the difference between a trade offer and a trick".
#:
#: Both of those come in a `-unclear` form too, and the pair matters more than either.
#: They used to be bare sign tests on a paired estimate too noisy to have a sign: across
#: three seeds on the live leagues the same forty trades disagreed with themselves 35-75%
#: of the time about `counterparty-loses` and 42-78% about `harmful`. "Cannot tell" is a
#: different sentence from "falls", and a reader deciding whether to send a trade offer
#: needs to know which one they are being handed.
_CAVEAT_TAGS: Mapping[str, str] = {
    "counterparty-loses": "a counterparty's simulated title odds FALL under this; "
    "the points gate it passed is not the same test",
    "counterparty-loses-unclear": "a counterparty's simulated title odds read slightly "
    "down, but not by enough to tell from noise -- the points gate it passed is a "
    "different test and this one did not resolve",
    "harmful": "the simulation disagrees with the screen about your own side",
    "harmful-unclear": "the simulation does not confirm the screen about your own side, "
    "and cannot separate the difference from noise",
    "partially-unpriced": "later weeks have no closing line yet and use the "
    "projection-only fit; those numbers will move",
    "not-streamable": "the fitted within-week spread at this position is below the "
    "streamable floor -- treat this as a tie-break, not an edge",
    "action-suppressed": "a week-one transaction was suppressed because the position "
    "is not streamable",
    "unilateral": "priced as an add with no corresponding drop",
    "roster_size": "the post-move roster is over its size limit",
}


def verdict_for(rec: Recommendation, *, surface: str = "") -> Verdict:
    """The right significance test for the surface that produced this recommendation.

    `decide/trades.py` has already applied the selection-adjusted threshold and encoded
    the answer in `confidence`, so a trade is judged on that; everything else is judged
    on the two-sigma test with the zero-effect hole closed. Getting this wrong in either
    direction is how a report lies: a two-sigma label on a search winner over-promises,
    and treating an exact zero as significant recommends doing nothing at high
    confidence.

    Two rules here are corrections to an earlier version of this function that both
    fired on live data.

    **A negative delta is a measured harm, not noise.** `_confidence` in
    `decide/trades.py` returns `"low"` for a confirmed trade whose measured title delta
    is <= 0 as well as for one inside its error, and the live example it was written
    against is `-0.80pp +/- 0.31` -- 2.6 sigma of *loss*. Labelling that "inside the
    selection-adjusted error" is a false statement about a number the simulation
    resolved perfectly well.

    **`confidence == "low"` is a surface refusing to stand behind its own number, on
    every surface and not only on trades.** `decide/streaming.py` sets it together with
    `not-streamable` when the fitted within-week spread at a position is below the floor
    -- the kicker case, R^2 = 0.022, where its own docstring says a surface that dresses
    that up as a recommendation is worse than no surface. The plan's paired standard
    error is small and the two-sigma test passes it happily: on the live Type shi kicker
    grid the plan came back +0.85pp +/- 0.46, and a seed that put it at +0.85 +/- 0.35
    would have been rendered `act`. The producing surface already said no; this must not
    overrule it with a test the surface does not use.
    """
    if rec.delta_title == 0.0 or _NULL_TAGS & set(rec.tags):
        return Verdict("null", "nothing measured: this move changes no roster")
    if rec.delta_title < 0.0:
        # A negative delta is never actionable, on any surface and at any confidence.
        # Only whether it is resolved varies.
        if rec.stderr > 0 and abs(rec.delta_title) <= 2.0 * rec.stderr:
            return Verdict("noise", "a loss, but inside two standard errors")
        return Verdict("harm", "measured as a LOSS of title probability -- do not do this")
    if surface == "trade":
        if rec.confidence == "low":
            return Verdict("noise", "inside the selection-adjusted error for a search winner")
        return Verdict("act", "clears the selection-adjusted threshold")
    if rec.stderr > 0 and abs(rec.delta_title) <= 2.0 * rec.stderr:
        return Verdict("noise", "inside two standard errors")
    if rec.confidence == "low":
        why = sorted(_LOW_CONFIDENCE_TAGS & set(rec.tags))
        return Verdict(
            "noise",
            "clears two standard errors, but the surface that produced it rates it "
            + "confidence='low'"
            + (f" ({', '.join(why)})" if why else ""),
        )
    return Verdict("act", "clears two standard errors")


def caveats_for(tags: Sequence[str]) -> list[str]:
    """The producing surface's own warnings, in English, off `Recommendation.tags`.

    These were previously carried into the payload as raw tags and rendered nowhere, so
    a live trade tagged `counterparty-loses` -- the simulation says a counterparty's
    title odds *fall* under a trade the points gate called Pareto-improving -- became the
    weekly headline for Blacksburg with no mention of it anywhere in the output.
    """
    return [_CAVEAT_TAGS[t] for t in sorted(set(tags) & set(_CAVEAT_TAGS))]


def rec_payload(
    rec: Recommendation,
    names: Mapping[int, str],
    *,
    team_id: int | None = None,
    surface: str = "",
) -> dict[str, Any]:
    """One `core.Recommendation` as JSON, with the players named and a verdict attached."""
    verdict = verdict_for(rec, surface=surface)
    moved = rec.move.players if team_id is not None else ()
    receive = [p.player_id for p in moved if p.to_team == team_id]
    send = [p.player_id for p in moved if p.from_team == team_id]
    return {
        "kind": rec.move.kind.value,
        "league_id": rec.move.league_id,
        "delta_title": rec.delta_title,
        "delta_points": rec.delta_points,
        "stderr": rec.stderr,
        "z": abs(rec.delta_title) / rec.stderr if rec.stderr > 0 else None,
        "significant": verdict.significant,
        "verdict": verdict.kind,
        "verdict_note": verdict.note,
        "leverage": rec.leverage,
        "confidence": rec.confidence,
        "tags": list(rec.tags),
        "rationale": rec.rationale,
        "receive": [{"player_id": p, "name": names.get(p, str(p))} for p in receive],
        "send": [{"player_id": p, "name": names.get(p, str(p))} for p in send],
        "players": [
            {
                "player_id": p.player_id,
                "name": names.get(p.player_id, str(p.player_id)),
                "from_team": p.from_team,
                "to_team": p.to_team,
            }
            for p in rec.move.players
        ],
    }


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def pp(value: float | None, digits: int = 2) -> str:
    """A probability difference in signed percentage points. The unit of everything."""
    return "-" if value is None else f"{value * 100:+.{digits}f}pp"


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def signed(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def slot_label(slot_id: int, eligibility: Mapping[int, frozenset[int]]) -> str:
    """A starting slot's name, derived from what it accepts rather than hard-coded.

    Slot ids and position ids collide at 4 and 15, so a table of slot names is a table of
    chances to print WR where TE belongs. The eligible position set is already on the
    state and cannot be wrong.
    """
    eligible = sorted(eligibility.get(slot_id, frozenset()))
    if not eligible:
        return str(slot_id)
    if len(eligible) == 1:
        return POSITION_ABBREV.get(eligible[0], str(eligible[0]))
    return "FLEX"


def _style_for(verdict: str) -> str:
    return {"act": "", "noise": "dim italic", "null": "dim", "harm": "bold red"}.get(verdict, "")


def _cell(text: str, verdict: str) -> Text:
    """A table cell de-emphasised when its row's effect is inside its own error."""
    return Text(text, style=_style_for(verdict))


def _marker(verdict: str) -> str:
    """The one-character reason a number should not be acted on, or nothing."""
    return {"noise": " ~", "null": " -", "harm": " !"}.get(verdict, "")


def _legend() -> Text:
    return Text(
        "  ~ effect is inside its own error (shown, not acted on)   "
        "- nothing measured   ! measured in the WRONG direction   "
        "deltas are signed, pp = percentage points of title",
        style="dim",
    )


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def tag_value(rec: Recommendation, prefix: str) -> str:
    """Read one `key:value` tag off a recommendation, or `-`.

    `core.Recommendation` carries a fixed set of fields, so every surface that has more
    to say says it in `tags`; the waiver board's add/drop/position names arrive that way.
    Read here rather than through `decide/waivers._tag` because a private name in another
    module is not a contract, and this is three lines.
    """
    for tag in rec.tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return "-"


def _section(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one section, or record why it could not run.

    A trade search that blows up must not cost the user his waiver board. Every section
    is independent and every failure is reported in place, with the exception type kept
    so the message says what actually happened rather than "an error occurred".
    """
    try:
        return {"ok": True, **fn()}
    except Exception as err:
        log.info("section %s failed", name, exc_info=True)
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


def champ_stderr(p: float, n_sims: int) -> float:
    """Monte Carlo error on one team's championship probability.

    The champion indicator is Bernoulli over `n_sims` independent simulated seasons, so
    this is `sqrt(p(1-p)/n)` and nothing subtler. At the 4,000 simulations `fq odds`
    runs by default a 13% favourite carries +/-0.53pp and a 2% also-ran +/-0.23pp,
    which is the entire reason the ranking below needs an error column: the table prints
    two decimals it does not have.
    """
    if n_sims <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n_sims)


def _rank_separated(p_hi: float, p_lo: float, n_sims: int) -> bool:
    """Whether two adjacent rows are actually in that order, on this draw.

    The two champion indicators are *mutually exclusive* -- one team wins the league --
    so `E[X_hi * X_lo] = 0`, `Cov = -p_hi*p_lo`, and the paired variance is
    `V_hi + V_lo + 2*p_hi*p_lo = p_hi + p_lo - (p_hi - p_lo)^2` over `n`. Note the sign:
    the exclusivity makes the difference *harder* to resolve than an independent pair
    would be, not easier, because a simulation that gives one of them the title
    necessarily takes it from the other. Using the naive `hypot` of the two Bernoulli
    errors would understate the bar by about 6% on a live adjacent pair and call some
    orderings real that are not.

    On the live 14-team board at 4,000 sims most adjacent pairs fail this, and the seeds
    agree: re-running Wine Wednesday under five seeds moved ranks 4 through 11 by up to
    three places each and flipped the user's own team between 13th and 14th.
    """
    if n_sims <= 0:
        return False
    var = (p_hi + p_lo - (p_hi - p_lo) ** 2) / n_sims
    if var <= 0:
        return p_hi > p_lo
    return (p_hi - p_lo) > 2.0 * math.sqrt(var)


def odds_payload(ws: Workspace, cfg: registry.LeagueConfig) -> dict[str, Any]:
    """Every team's championship odds, plus the facts ESPN already knows.

    The probabilities come from `pipeline.championship_table`, which **floors an unfilled
    starting slot at the wire**, as every recommendation surface already did. It used to
    score that slot at zero, which made this the one surface in the system answering a
    different question from the rest -- a gap disclosed only as a footer string here and
    impossible to reconcile from the output.

    The remaining disagreement with `fq waivers` is the DRAW, not the floor: that board
    runs on a widened panel so free agents have columns. Compare deltas, not levels.

    **The ranking is mostly Monte Carlo noise and the payload now says so.** This table
    is the first thing `fq weekly` prints and the only surface here that had no
    significance treatment at all -- it published `12.95%` against `11.30%` against
    `11.10%` with no error and an implied strict order. `championship_stderr` and
    `separated_from_next` are computed per row and `n_ranks_separated` counts how many
    adjacent pairs survive; on the live leagues that is a small minority of them.
    """
    sim = ws.sim(cfg)
    read, wanted = ws.variant(cfg)
    team_id = ws.my_team_id(cfg)
    rows = pipeline.championship_table(sim)
    espn = ws.teams(cfg)
    teams = []
    for rank, row in enumerate(rows, start=1):
        tid = int(row["team_id"])
        team = espn.get(tid)
        teams.append(
            {
                "rank": rank,
                "team_id": tid,
                "name": row["name"],
                "is_me": tid == team_id,
                "championship": float(row["championship"]),
                "playoffs": float(row["playoffs"]),
                "bye": float(row["bye"]),
                "expected_wins": float(row["expected_wins"]),
                "wins": team.record.wins if team else None,
                "losses": team.record.losses if team else None,
                "ties": team.record.ties if team else None,
                "points_for": team.record.points_for if team else None,
                "espn_projected_rank": team.current_projected_rank if team else None,
                "championship_stderr": champ_stderr(float(row["championship"]), sim.n_sims),
            }
        )
    for a, b in zip(teams, teams[1:], strict=False):
        a["separated_from_next"] = _rank_separated(a["championship"], b["championship"], sim.n_sims)
    if teams:
        teams[-1]["separated_from_next"] = None
    separated = sum(1 for t in teams if t.get("separated_from_next"))
    mine = next((t for t in teams if t["is_me"]), None)
    return {
        "league_id": cfg.league_id,
        "season": cfg.season,
        "name": sim.state.name,
        "size": sim.state.size,
        "week": sim.state.weeks[0] if sim.state.weeks else None,
        "team_id": team_id,
        "team_name": mine["name"] if mine else "",
        "n_sims": sim.n_sims,
        "corpus_variant": read,
        "corpus_variant_requested": wanted,
        "baseline": "championship_table (unfilled slot floored at the wire)",
        "teams": teams,
        "my_rank": mine["rank"] if mine else None,
        "my_championship": mine["championship"] if mine else None,
        "my_championship_stderr": mine["championship_stderr"] if mine else None,
        "my_playoffs": mine["playoffs"] if mine else None,
        "n_ranks_separated": separated,
        "n_ranks": max(len(teams) - 1, 0),
        "ranking_note": (
            f"{separated} of {max(len(teams) - 1, 0)} adjacent pairs are separated by more "
            f"than twice the paired Monte Carlo error at {sim.n_sims:,} simulations. The rest "
            "of this order is noise; raise --sims to resolve it."
        ),
    }


def leverage_payload(ws: Workspace, cfg: registry.LeagueConfig) -> dict[str, Any]:
    """How much a projected point is worth in each remaining matchup.

    Often the single most valuable line in the report, and the only one that can say
    *this week barely matters*. `leverage` is `core.leverage`: the marginal win
    probability per point relative to a coin flip, so 1.00 means every point lands and
    0.10 means the game is decided and the lineup call is not worth the click.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    engine = ws.engine(cfg)
    names = {f.team_id: f.name for f in sim.state.franchises}
    rows = [
        {
            "week": r.week,
            "matchup_period": r.matchup_period,
            "weeks_in_matchup": 1,
            "opponent_id": r.opponent_id,
            "opponent": names.get(r.opponent_id, str(r.opponent_id)),
            "margin": r.margin,
            "sd_diff": r.sd_diff,
            "win_probability": r.win_probability,
            "leverage": r.leverage,
            "decided": r.decided,
            "points_per_win_pct": r.points_per_win_pct,
        }
        for r in engine.week_leverage(team_id)
    ]
    return {
        "league_id": cfg.league_id,
        "name": sim.state.name,
        "team_id": team_id,
        "mean_leverage": engine.mean_leverage(team_id),
        "this_week": rows[0] if rows else None,
        "least_leveraged": min(rows, key=lambda r: r["leverage"]) if rows else None,
        "weeks": rows,
    }


def lineup_payload(
    ws: Workspace, cfg: registry.LeagueConfig, *, week: int | None = None
) -> dict[str, Any]:
    """The start/sit call, measured against the lineup that is actually set.

    `current` is read live off ESPN so the deltas answer "what does changing my lineup
    buy me" rather than "what would the optimum have been". When the roster cannot be
    read the surface still runs, and `current_known` says the deltas are against a
    hypothetical.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    current = ws.current_starters(cfg)
    names = ws.names(cfg)
    advice = lineups_mod.advise_sim(
        sim, team_id=team_id, week=week, current=list(current) if current else None
    )
    eligibility = sim.state.slot_eligibility

    def _lineup(option: Any) -> list[dict[str, Any]]:
        return [
            {
                "slot_id": int(s),
                "slot": slot_label(int(s), eligibility),
                "player_id": int(p),
                "name": n,
            }
            for s, p, n in zip(option.slot_ids, option.player_ids, option.names, strict=True)
        ]

    changes = [
        {
            "slot_id": s.slot_id,
            "slot": slot_label(s.slot_id, eligibility),
            "out_player_id": s.out_player_id,
            "out": s.out_name,
            "in_player_id": s.in_player_id,
            "in": s.in_name,
            "d_mean": s.d_mean,
            "d_sd": s.d_sd,
        }
        for s in advice.swaps
    ]
    return {
        "league_id": cfg.league_id,
        "name": sim.state.name,
        "team_id": team_id,
        "team_name": advice.team_name,
        "week": advice.week,
        "threshold_kind": advice.kind.value,
        "target": advice.target,
        "margin": advice.margin,
        "sd_diff": advice.sd_diff,
        "z": advice.z,
        "leverage": advice.leverage,
        "current_known": current is not None,
        "unpriced_current": [int(p) for p in advice.unpriced_current],
        "opponent": (
            {
                "team_id": advice.opponent.team_id,
                "name": advice.opponent.name,
                "mean": advice.opponent.mean,
                "sd": advice.opponent.sd,
            }
            if advice.opponent
            else None
        ),
        "recommended": _lineup(advice.recommended),
        "baseline": _lineup(advice.baseline),
        "changes": changes,
        "n_changes": len(changes),
        "points_lineup_mean": advice.points_lineup.mean,
        "win_prob_lineup_mean": advice.win_prob_lineup.mean,
        "delta_win_prob": advice.delta_win_prob,
        "delta_win_prob_stderr": advice.delta_win_prob_stderr,
        "noise_floor": advice.noise_floor,
        "points_sacrifice": advice.points_sacrifice,
        "guard": advice.guard,
        "differ": advice.differ,
        "significant": advice.significant,
        "n_lineups": advice.n_lineups,
        "sd_independent": advice.sd_independent,
        "stacks": [
            {
                "name": e.name,
                "opponent_name": e.opponent_name,
                "rho": e.rho,
                "modelled": e.modelled,
            }
            for e in advice.stacks
        ],
        # `LineupAdvice.significant` is stricter than `Recommendation.significant`: an
        # already-optimal lineup produces two bit-identical simulated arms, so the
        # difference and its error are both exactly zero.
        "recommendation": {
            **rec_payload(advice.recommendation, names, team_id=team_id, surface="lineup"),
            "significant": advice.significant,
            "verdict": "act" if advice.significant else ("null" if not changes else "noise"),
        },
    }


def _rankings_payload(board: Any, matched: bool, weight: float) -> dict[str, Any] | None:
    """What board a surface was priced against, or None when it ran on ESPN alone."""
    if board is None:
        return None
    return {
        "kind": board.kind,
        "scoring": board.scoring,
        "matches_league_scoring": matched,
        "n": board.n,
        "weight": weight,
        "file": board.path.name,
        "unverified": (
            "This board ships without a measured verdict: no historical boards exist to "
            "score it against. `data.etr.archive` is accumulating them."
        ),
    }


def waivers_payload(
    ws: Workspace, cfg: registry.LeagueConfig, *, limit: int = 10, week: int | None = None
) -> dict[str, Any]:
    """The waiver board and the priority threshold a claim has to clear.

    The threshold is the point of it. All three of the user's leagues run rolling waiver
    priority, so a claim costs a queue position rather than money, and the right question
    is not "is this player good" but "is he better than the option of claiming somebody
    later". `decide/waivers.py` solves that by backward induction; this reports the
    number and which candidates clear it.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    names = ws.names(cfg)
    rankings, rankings_match = ws.rankings(cfg)
    report = waivers_mod.waiver_board(sim, team_id=team_id, week=week, rankings=rankings)
    on_waivers = {a.name: a.on_waivers for a in report.free_agents}

    def _row(rec: Recommendation) -> dict[str, Any]:
        price = waivers_mod.price_of(rec)
        body = rec_payload(rec, names, team_id=team_id, surface="waiver")
        body.update(
            {
                "add": tag_value(rec, "add:"),
                "position": tag_value(rec, "pos:"),
                "drop": tag_value(rec, "drop:"),
                "bracket_title": price.bracket_title,
                "bracket_stderr": price.bracket_stderr,
                "agrees": price.agrees,
                # `ClaimPrice.significant`, not `Recommendation.significant`: a free
                # agent strictly below the wire floor adds exactly nothing in every
                # simulation, so both the effect and its error are identically zero.
                "significant": price.significant,
                "verdict": (
                    "act" if price.significant else ("null" if not rec.delta_title else "noise")
                ),
                # A player who costs no waiver claim has no threshold to clear. Reporting
                # one against him is the presentation half of the bug in `waiver_board`:
                # a first-come free agent was being told to wait for a Wednesday run.
                "on_waivers": on_waivers.get(tag_value(rec, "add:"), True),
                "cost": (
                    "waiver priority"
                    if on_waivers.get(tag_value(rec, "add:"), True)
                    else "free"
                ),
                "clears_threshold": (
                    rec.delta_title >= report.threshold
                    if on_waivers.get(tag_value(rec, "add:"), True)
                    else rec.delta_title > 0.0
                ),
                # The `clears?` column was a hard `>=` against a threshold, printed as a
                # fact, and the claim waterfall is cut on it. Live, the margin over the
                # threshold is inside the row's own error on the marginal candidate in
                # two of the three leagues -- Blacksburg's fifth claim clears by
                # +0.003pp against +/-0.024pp and its sixth misses by -0.004pp -- so
                # "submit 5" is really "submit 4, and the fifth is a coin flip". This is
                # a LOWER bound on that uncertainty: the threshold itself is estimated by
                # backward induction and carries an error this cannot see.
                "clears_margin": rec.delta_title - report.threshold,
                "clears_certain": abs(rec.delta_title - report.threshold) > 2.0 * rec.stderr,
                # The analyst's own rank comparison and note, when the board has one.
                # Derived from the board, not from `delta_title`, which already carries
                # the tilt -- see `waivers._board_tags`.
                "board": tag_value(rec, "board:"),
                "note": tag_value(rec, "note:"),
                # "+0.0/+16.3": what dropping him costs as lineups are actually set,
                # and what he would have been worth to someone who knew which weeks to
                # start him. Reported, never charged -- see `waivers._ceiling_tags`.
                "drop_ceiling": tag_value(rec, "ceiling:"),
            }
        )
        return body

    board = [_row(r) for r in report.board[:limit]]
    claims = [_row(r) for r in report.claims]
    return {
        "league_id": cfg.league_id,
        "name": sim.state.name,
        "team_id": team_id,
        "team_name": report.team_name,
        "week": report.week,
        "uses_faab": report.uses_faab,
        "priority": report.priority,
        "priority_known": report.priority_known,
        "budget": report.budget,
        "threshold": report.threshold,
        "baseline_title": report.baseline_title,
        "baseline": "waiver board (unfilled slot streams a replacement)",
        "title_per_point": report.title_per_point,
        "title_per_point_stderr": report.title_per_point_stderr,
        "week_leverage": report.week_leverage,
        "sd_diff": report.sd_diff,
        "n_free_agents": len(report.free_agents),
        # What the league actually looks like right now. "809 of these cost you nothing"
        # is the answer to "why am I being told to wait until Tuesday", and it is the
        # first thing a reader should see next to a threshold.
        "n_on_waivers": report.n_on_waivers,
        "n_free_agents_available": report.n_free_agents_available,
        "rankings": _rankings_payload(rankings, rankings_match, waivers_mod.DEFAULT_BOARD_WEIGHT),
        # Per single-body slot: what the seat is worth taking the best available every
        # week against holding the best rosterable body. The gap is non-negative by
        # construction, so only its size across positions carries information -- D/ST
        # runs ~37 points a season against ~13 at K and TE. See `stream_advantage`.
        "stream_advantage": [
            {
                "slot": slot,
                "position": POSITION_ABBREV.get(
                    next(iter(sim.state.slot_eligibility.get(slot, ())), 0), str(slot)
                ),
                "stream_points": stream,
                "hold_points": hold,
                "hold_player": who,
                "gap": stream - hold,
            }
            for slot, (stream, hold, who) in sorted(report.stream_advantage.items())
        ],
        "board": board,
        "claims": claims,
        "free_adds": [_row(r) for r in report.free_adds],
        "blocks": [_row(r) for r in report.blocks],
        "hold": rec_payload(report.hold, names, team_id=team_id, surface="waiver"),
        "best": rec_payload(report.best, names, team_id=team_id, surface="waiver"),
        "any_claim": bool(claims),
        "any_action": bool(claims) or bool(report.free_adds),
        "waterfall_note": (
            "A losing claim is free -- winning moves you to the back of the queue and "
            "losing leaves you where you were -- so submit every candidate above the "
            "threshold, in order."
        )
        if claims
        else "",
    }


def trades_payload(
    ws: Workspace, cfg: registry.LeagueConfig, *, limit: int = 5, min_gain: float = 0.0
) -> dict[str, Any]:
    """Confirmed Pareto trades, best first, from this team's side of the table.

    `find_trades` gates on a strict Pareto improvement in playoff-weighted points and
    then ranks on the paired title simulation. Its own docstring is blunt that the gate
    passes legs a counterparty gains half a point from, which is not the same thing as a
    trade a human accepts; `--min-gain` raises the gate and is plumbed through for that.

    **The point estimate on the top row is biased upward and the report has to say so.**
    `decide/trades.selection_threshold` corrects the *label* -- how many sigma a winner
    has to clear -- and nothing corrects the *number*, because nothing can without a
    second independent draw. It is the maximum of `n_found` positively correlated noisy
    paired estimates, and its own module measured what that costs: a top trade at
    +1.12pp on one seed is +0.53pp at ten times the simulations and +0.18pp on another
    seed of the same size. Re-running Blacksburg's live board under five seeds moved the
    same trade -- Hubbard and Waddle for Mason and Corum -- between +1.10pp and +1.83pp
    and flipped its verdict between `act` and `noise` three times to two. So `+/-` here
    is the paired Monte Carlo error on one draw and is **not** the uncertainty on the
    trade being worth what it says; `selection_note` says that in the output rather than
    in a docstring, and `expected_shrinkage` is the honest reading of the top row.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    names = ws.names(cfg)
    franchises = {f.team_id: f.name for f in sim.state.franchises}
    rankings, rankings_match = ws.rankings(cfg)
    recs = trades_mod.find_trades(sim, for_team=team_id, min_gain=min_gain, rankings=rankings)
    notes = rankings.comments() if rankings is not None else {}
    rows = []
    for rec in recs[:limit]:
        body = rec_payload(rec, names, team_id=team_id, surface="trade")
        partners = sorted({t for t in rec.move.teams if t != team_id})
        body["partners"] = [{"team_id": t, "name": franchises.get(t, str(t))} for t in partners]
        body["caveats"] = caveats_for(rec.tags)
        # The arbitrage, as two numbers the reader can hold side by side: what the deal
        # is worth to me by the analyst board, and how much better the other side thinks
        # it is doing by the projections on their own screen. `spread` is the second
        # minus the first on their side only; see `TradeEvaluation.spread`.
        # `tag_value` returns "-" for a missing tag, and "-" is truthy: reading it
        # through `float()` raised `ValueError` on every board-less run, which is every
        # fresh checkout, every `--no-rankings`, and the dashboard's own trades route.
        # Gate on the board rather than on the sentinel.
        spread = tag_value(rec, "spread:") if rankings is not None else "-"
        body["spread"] = float(spread) if spread and spread != "-" else None
        body["mispriced"] = "mispriced" in rec.tags
        # How many other ways there are to end up with exactly these players for exactly
        # this price, differing only in who stands in the middle. `same_return` marks
        # those alternatives; the row that leads a family is the easiest one to get
        # signed. See `trades.order_routes`.
        routes = tag_value(rec, "routes:")
        body["routes"] = int(routes) if routes.isdigit() else 0
        body["same_return"] = "same-return" in rec.tags
        body["notes"] = {
            names.get(p.player_id, str(p.player_id)): notes[p.player_id]
            for p in rec.move.players
            if p.player_id in notes
        }
        rows.append(body)
    n_found = len(recs)
    return {
        "league_id": cfg.league_id,
        "name": sim.state.name,
        "team_id": team_id,
        "n_found": n_found,
        "min_gain": min_gain,
        "rankings": _rankings_payload(
            rankings, rankings_match, trades_mod.DEFAULT_RANKINGS_WEIGHT
        ),
        # `delta_title` on every row is priced on the re-dealt projections when a board
        # is present, and the paired baseline inside `find_trades` is priced the same
        # way -- so the deltas are internally consistent, but the ABSOLUTE title level
        # behind them is not `fq odds`'s. That is the shape of the bug `411ed47` fixed,
        # so it is stated here and no tilted level is printed beside the odds table.
        "priced_on": "analyst board" if rankings is not None else "espn projections",
        "significance_test": "selection-adjusted (decide.trades.selection_threshold)",
        "selection_note": (
            f"dTitle on the top row is the MAXIMUM of {n_found} noisy paired estimates, so it "
            "is biased upward; the +/- is this draw's Monte Carlo error and not the "
            "uncertainty on the trade being worth that. Re-run with --seed to see how far "
            "it moves before you propose anything."
        )
        if n_found > 1
        else "",
        "trades": rows,
    }


def stream_payload(
    ws: Workspace,
    cfg: registry.LeagueConfig,
    *,
    position_id: int = DST,
    plan_weeks: int = 8,
) -> dict[str, Any]:
    """This week's streaming action, and the rest-of-season plan that justifies it.

    `delta_title` prices the WHOLE plan, not the move -- on all three real D/ST grids
    week one's action is a hold, so reading it as the value of the move says doing
    nothing is worth several points of title probability. `decide/streaming.py` tags that
    `no-action-this-week` and this keeps the tag rather than flattening it.

    The week-by-week table is re-derived through the same public helpers `recommend`
    composes, because `recommend` returns only the `Recommendation`. The first week of
    the re-derived plan is checked against the recommendation's own move, and the table
    is dropped rather than shown if they disagree -- a plan that does not match the
    number beside it is worse than no plan.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    names = ws.names(cfg)
    rec = streaming_mod.recommend(sim, position_id=position_id)
    body = rec_payload(rec, names, team_id=team_id, surface="stream")
    body["position"] = POSITION_ABBREV.get(position_id, str(position_id))
    body["caveats"] = caveats_for(rec.tags)
    full = _stream_plan_rows(sim, position_id, rec, limit=None)
    body["plan"] = full[:plan_weeks]
    body["plan_source"] = "rederived" if full else "unavailable"
    body["plan_summary"] = _plan_summary(full, shown=len(body["plan"]))
    body["action"] = (
        "hold" if "no-action-this-week" in rec.tags else rec.move.kind.value.replace("_", "/")
    )
    return {
        "league_id": cfg.league_id,
        "name": sim.state.name,
        "team_id": team_id,
        "stream": body,
    }


def _plan_summary(rows: Sequence[Mapping[str, Any]], *, shown: int) -> dict[str, Any]:
    """Where the streaming plan's value actually sits, from the plan table itself.

    Three facts the report used to print a single number over, all of which change what
    the number means and none of which needs a second simulation.

    *How much of it is this week.* Week one's gain over holding is +0.00 on all three
    live D/ST grids, because week one's action is a hold. The `+2.77pp` beside it prices
    a seventeen-week plan and the executable-today part of it is zero.

    *How much of it rests on a line nobody has posted.* `decide/streaming.py` measured
    that only 13.4-14.9 of ~43 plan points sit in weeks a bookmaker has priced, and that
    pricing just those gives `+0.45pp +/- 0.21` -- a tenth of the headline and barely
    two sigma. `opponent_priced` is on every row, so the split is arithmetic.

    *How many transactions it assumes.* A plan worth +3.8pp that requires fifteen
    successful waiver adds over seventeen weeks is not the same offer as one requiring
    two, and the queue ranks both on the same axis.
    """
    if not rows:
        return {}
    gains = [float(r.get("gain") or 0.0) for r in rows]
    priced = sum(g for g, r in zip(gains, rows, strict=True) if r.get("opponent_priced"))
    total = sum(gains)
    moves = sum(1 for r in rows if r.get("add"))
    return {
        "weeks": len(rows),
        "weeks_shown": shown,
        "total_gain_points": total,
        "priced_gain_points": priced,
        "unpriced_gain_points": total - priced,
        "unpriced_share": (total - priced) / total if total else 0.0,
        "week_one_gain_points": gains[0],
        "n_acquisitions": moves,
    }


def _stream_plan_rows(
    sim: Any, position_id: int, rec: Recommendation, *, limit: int | None
) -> list[dict[str, Any]]:
    """The plan `streaming.recommend` solved, re-solved for display only.

    Guarded end to end: a disagreement between this plan's week one and the
    recommendation's own move means the two were not solved from the same grid, and the
    only safe thing to do with a table that might describe a different plan is not to
    print it.
    """
    try:
        state = sim.state
        market = streaming_mod.market_schedule(state.season)
        ownership = {pid: f.team_id for f in state.franchises for pid in f.player_ids}
        grid = streaming_mod.build_grid(
            sim.outlooks,
            league_id=state.league_id,
            season=state.season,
            position_id=position_id,
            weeks=state.weeks,
            ownership=ownership,
            my_team_id=state.my_team_id,
            market=market,
        )
        kappa = max(1, len(grid.held_index))
        plan = streaming_mod.solve(grid, kappa=kappa)
        base = streaming_mod.hold_plan(grid)
        rows = streaming_mod.plan_table(grid, plan, base)
    except Exception as err:
        log.info("could not re-derive the streaming plan: %s", err)
        return []
    if rows and rec.move.lineup:
        started = next(iter(rec.move.lineup.values()))
        # `plan.start[0] < 0` is "leave the seat empty", and a negative index would wrap
        # to the last streamer and compare against the wrong player rather than disagree.
        here = grid.streamers[plan.start[0]].player_id if plan.start[0] >= 0 else None
        if here != started:
            log.info("re-derived streaming plan disagrees with the recommendation; dropping it")
            return []
    return list(rows) if limit is None else rows[:limit]


# --------------------------------------------------------------------------------------
# The weekly report
# --------------------------------------------------------------------------------------


def weekly_payload(
    ws: Workspace,
    cfg: registry.LeagueConfig,
    *,
    sections: Sequence[str] = WEEKLY_SECTIONS,
    limit: int = 6,
) -> dict[str, Any]:
    """One league's whole picture. Every section fails independently."""
    chosen = [s for s in WEEKLY_SECTIONS if s in set(sections)]
    out: dict[str, Any] = {
        "league_id": cfg.league_id,
        "season": cfg.season,
        "name": cfg.name,
        "ok": True,
    }
    try:
        sim = ws.sim(cfg)
        out["name"] = sim.state.name
        out["team_id"] = ws.my_team_id(cfg)
        out["week"] = sim.state.weeks[0] if sim.state.weeks else None
        out["n_sims"] = sim.n_sims
        read_variant, want_variant = ws.variant(cfg)
        out["corpus_variant"] = read_variant
        out["corpus_variant_requested"] = want_variant
    except Exception as err:
        log.info("league %s could not be built", cfg.league_id, exc_info=True)
        return {**out, "ok": False, "error": f"{type(err).__name__}: {err}"}

    builders: dict[str, Callable[[], dict[str, Any]]] = {
        "odds": lambda: odds_payload(ws, cfg),
        "leverage": lambda: leverage_payload(ws, cfg),
        "lineup": lambda: lineup_payload(ws, cfg),
        "waivers": lambda: waivers_payload(ws, cfg, limit=limit),
        "trades": lambda: trades_payload(ws, cfg, limit=limit),
        "stream": lambda: stream_payload(ws, cfg),
    }
    for name in chosen:
        out[name] = _section(name, builders[name])
    out["headline"] = headline(out)
    out["actions"] = [a.to_dict() for a in actions_from(cfg, out)]
    return out


def headline(payload: Mapping[str, Any]) -> str:
    """The one line to read if you read nothing else, composed from measured numbers.

    A hold gets said plainly and quantitatively: what the best available move was worth,
    what it had to clear, and what a point is worth this week. That sentence is the
    honest output for most weeks in most leagues, and dressing it up as a recommendation
    is the failure mode this whole module is meant to avoid.
    """
    best = _best_action(payload)
    lev = _get(payload, "leverage", "this_week") or {}
    leverage_value = lev.get("leverage")
    week_note = ""
    if leverage_value is not None:
        if leverage_value < DECIDED_LEVERAGE:
            week_note = (
                f" This week is close to decided (leverage {leverage_value:.2f} against "
                f"{lev.get('opponent', 'your opponent')}), so a marginal upgrade is worth "
                f"about {lev.get('points_per_win_pct', 0.0):.2f}pp of the game per point."
            )
        else:
            week_note = (
                f" This week's matchup is live (leverage {leverage_value:.2f}); a projected "
                f"point is worth {lev.get('points_per_win_pct', 0.0):.2f}pp of it."
            )

    # `actionable` as well as `significant`: a claim can be a real measured gain and
    # still be one you should not make, because it does not clear the cost of spending
    # priority. Leading the report with it would recommend exactly what the threshold
    # exists to prevent.
    if best is not None and best.significant and best.actionable:
        # The caveats are part of the headline, not a footnote to it. The live case this
        # was written for: Blacksburg's lead recommendation is a three-team trade tagged
        # `counterparty-loses`, and the sentence used to end at "+1.75pp (+/-0.45pp)".
        warn = "".join(f" Caveat: {c}." for c in best.caveats)
        return (
            f"{best.headline} Worth {pp(best.delta_title)} "
            f"(+/-{best.stderr * 100:.2f}pp).{warn}{week_note}"
        )

    threshold = _get(payload, "waivers", "threshold")
    top = _get(payload, "waivers", "board") or []
    rate = _get(payload, "waivers", "title_per_point")
    # Two different reasons to hold, and they are not interchangeable. Either nothing
    # measured clears its own error, or something did and is still not worth what it
    # costs -- a claim that would spend a waiver priority worth more than the claim.
    # Two different reasons, and the second one has to name WHICH gain and WHY it is not
    # executable, because the largest measured number in the league is routinely a
    # rest-of-season streaming plan whose week-one action is a hold. "The one measured
    # gain here is not worth what it costs" was written for the waiver case and read as a
    # claim about the wire even when the number it referred to was a D/ST plan.
    if best is None or not best.significant:
        lead = "Hold -- nothing here clears its own error."
    else:
        why = f" ({'; '.join(best.blockers)})" if best.blockers else ""
        lead = (
            f"Hold -- the largest measured gain here is the {best.surface} at "
            f"{pp(best.delta_title)}, and it is not something to execute today{why}."
        )
    parts = [lead]
    if top and threshold is not None:
        parts.append(
            f"The best claim on the wire ({top[0].get('add', '?')}) is worth "
            f"{pp(top[0].get('delta_title'))} against a {threshold * 100:.3f}pp cost of "
            f"spending waiver priority."
        )
    elif threshold is not None:
        parts.append(f"Nothing on the wire clears the {threshold * 100:.3f}pp priority threshold.")
    if _get(payload, "lineup", "n_changes") == 0:
        parts.append("Your lineup is already the one to start.")
    if rate:
        parts.append(f"A rest-of-season point is worth {rate * 100:.3f}pp of title here.")
    return " ".join(parts) + week_note


def _get(payload: Mapping[str, Any], section: str, *path: str) -> Any:
    """Read into a section that may have failed, without a wall of `.get` chains."""
    node = payload.get(section)
    if not isinstance(node, Mapping) or not node.get("ok", True):
        return None
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


# --------------------------------------------------------------------------------------
# The cross-league action queue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Action:
    """One thing to do, in one league, priced in the unit that crosses leagues.

    `delta_title` is the only field that is comparable between two leagues, which is the
    whole argument for the queue existing. `cost` and `deadline` are not comparable and
    are not ranked on -- they are what the user needs to actually execute the thing.
    """

    league_id: int
    league_name: str
    season: int
    team_id: int
    surface: str
    headline: str
    delta_title: float
    stderr: float
    leverage: float
    significant: bool
    verdict: str
    confidence: str
    cost: str
    deadline: str
    tags: tuple[str, ...] = ()
    rationale: str = ""
    #: Whether there is anything to execute today. A streaming plan worth +2.8pp whose
    #: first week is a hold is real and is not a thing to do this morning.
    actionable: bool = True
    #: Why it is not simply "do it", in the producing surface's own vocabulary.
    blockers: tuple[str, ...] = ()
    #: What is wrong with the NUMBER, as opposed to with executing it. A search winner's
    #: upward bias, a counterparty the simulation says loses, a plan resting on weeks no
    #: bookmaker has priced. These used to live in `tags` and render nowhere.
    caveats: tuple[str, ...] = ()

    @property
    def z(self) -> float | None:
        return abs(self.delta_title) / self.stderr if self.stderr > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_id": self.league_id,
            "league_name": self.league_name,
            "season": self.season,
            "team_id": self.team_id,
            "surface": self.surface,
            "headline": self.headline,
            "delta_title": self.delta_title,
            "stderr": self.stderr,
            "z": self.z,
            "leverage": self.leverage,
            "significant": self.significant,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "cost": self.cost,
            "deadline": self.deadline,
            "tags": list(self.tags),
            "rationale": self.rationale,
            "actionable": self.actionable,
            "blockers": list(self.blockers),
            "caveats": list(self.caveats),
        }


#: Only a tiebreak between actions of near-equal value: a free lineup change beats a
#: claim that spends waiver priority beats a trade that has to be negotiated. Never a
#: primary key -- the value is the value.
_COST_RANK = {"free": 0, "waiver priority": 1, "FAAB": 1, "negotiation": 2}


def rank_actions(actions: Sequence[Action], *, limit: int = 20) -> list[Action]:
    """The queue's own ordering: real effects first, then by size, then by what it costs.

    Anything inside its own error sorts below everything that is not, regardless of point
    estimate. That is deliberate and it changes the order: the largest number on a live
    board is routinely a noisy one, and putting it on top is exactly how a report gets
    someone to make a trade the simulation cannot actually distinguish from zero.
    """
    ordered = sorted(
        actions,
        key=lambda a: (
            not (a.significant and a.actionable),
            -a.delta_title,
            _COST_RANK.get(a.cost, 3),
            a.league_id,
        ),
    )
    return ordered[:limit]


#: Which surface costs what to execute, for the `cost` column. Not ranked on -- a
#: cheaper action does not become a better one -- but it is what the user needs to know
#: before he can do anything about the row above it.
_SURFACE_COST = {
    "waivers": "waiver priority",
    "waiver": "waiver priority",
    "streaming": "waiver priority",
    "stream": "waiver priority",
    "trades": "negotiation",
    "trade": "negotiation",
    "lineups": "free",
    "lineup": "free",
}
_SURFACE_DEADLINE = {
    "waiver priority": "Tuesday night",
    "negotiation": "league trade deadline",
    "free": "kickoff",
}


def _first_sentences(text: str, *, sentences: int = 1, limit: int = 140) -> str:
    """The surface's own opening sentences. Its words, not a template rewritten here.

    `sentences=2` exists for trades and is not cosmetic. `decide/trades.py` writes the
    pitch from the counterparties' side first -- "Zu's Saucy Calzone Squad gets Jordan
    Mason for Tyler Shough." -- and only the SECOND sentence says what the user himself
    receives. One sentence of that is a queue row describing somebody else's trade.
    """
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    if not parts:
        return ""
    head = ". ".join(parts[: max(sentences, 1)])
    head = head if head.endswith(".") else head + "."
    return head if len(head) <= limit else head[: limit - 1].rstrip() + "\u2026"


#: `edges.portfolio` names its surfaces after the modules; `verdict_for` names them
#: after the move. One map so the two vocabularies meet in exactly one place.
_PORTFOLIO_SURFACE = {
    "trades": "trade",
    "waivers": "waiver",
    "lineups": "lineup",
    "streaming": "stream",
}


def _action_from_item(item: Any, *, season: int) -> Action:
    """One `edges.portfolio.QueueItem` in this module's shape.

    `blockers` and `actionable` come straight off the item rather than being re-derived:
    the streaming surface's `no-action-this-week` is the one that matters, and it is the
    portfolio module that knows the tag vocabulary.

    **`significant` deliberately does not.** `QueueItem.significant` is
    `rec.delta_title != 0 and rec.significant` -- the plain two-sigma test -- which is
    the exact test this module exists to not apply to a trade that won a search. Taking
    it at face value made `fq queue` and `fq weekly` disagree about the same
    recommendation: Wine Wednesday's top trade is +1.03pp +/- 0.41 (z = 2.5), which
    `decide/trades.py` marks `confidence="low"` against a selection-adjusted threshold of
    3.2 sigma. The weekly report rendered it `~`; the queue -- the default path, and the
    one that prints "Do now:" -- rendered it as a thing to go and do. So the verdict is
    recomputed here through `verdict_for`, and a disagreement is recorded as a caveat
    rather than resolved silently in either direction.
    """
    rec = item.rec
    cost = _SURFACE_COST.get(item.surface, "free")
    surface = _PORTFOLIO_SURFACE.get(item.surface, item.surface)
    decided = verdict_for(rec, surface=surface)
    verdict = decided.kind
    caveats = caveats_for(rec.tags)
    # `QueueItem.n_considered` says how big a field this row won, when the module
    # publishes it. Read defensively: it is a sibling under active development, and the
    # caveat is worth having whenever it is available rather than only when it is
    # guaranteed. Same sentence the local merge attaches, so the two paths agree.
    n_considered = int(getattr(item, "n_considered", 1) or 1)
    if n_considered > 1:
        caveats.append(
            f"best of {n_considered} candidates: the point estimate is the maximum of "
            f"{n_considered} noisy paired draws and is biased upward"
        )
    if bool(getattr(item, "significant", False)) and not decided.significant:
        caveats.append(
            f"edges.portfolio's queue calls this row significant and the {surface} "
            f"surface's own test does not ({decided.note}); shown as not significant"
        )
    return Action(
        league_id=int(item.league_id),
        league_name=str(item.league_name),
        season=season,
        team_id=int(item.team_id),
        surface=str(item.surface),
        headline=(
            _first_sentences(rec.rationale, sentences=2, limit=220)
            if item.surface.startswith("trade")
            else _first_sentences(rec.rationale)
        )
        or rec.move.kind.value,
        delta_title=float(rec.delta_title),
        stderr=float(rec.stderr),
        leverage=float(rec.leverage),
        significant=decided.significant,
        verdict=verdict,
        confidence=str(rec.confidence),
        cost=cost,
        deadline=_SURFACE_DEADLINE.get(cost, "-"),
        tags=tuple(rec.tags),
        rationale=rec.rationale,
        actionable=bool(item.actionable),
        blockers=tuple(item.blockers),
        caveats=tuple(caveats),
    )


def _first_actionable(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Index of the first row that is both real and executable today, or None.

    Reported alongside the ranking rather than substituted for it. `edges.portfolio`
    ranks on the increment to `E[titles]` and deliberately does not sort holds down, so
    on the live leagues the top three rows are rest-of-season streaming plans whose week
    one is a hold: correct as a ranking of value, useless as a to-do list. This says
    which row is the to-do without reordering anything.
    """
    for i, row in enumerate(rows):
        if row.get("actionable", True) and row.get("significant"):
            return i
    return None


def portfolio_queue(
    ws: Workspace,
    configs: Sequence[registry.LeagueConfig],
    *,
    limit: int = 20,
    actionable_only: bool = False,
) -> dict[str, Any] | None:
    """The queue from `edges/portfolio.py`, or `None` when it cannot produce one.

    Preferred over this module's own merge whenever it is importable, and not out of
    politeness: `build_portfolio` puts every league on ONE shared NFL season under one
    seed, so simulation `s` in Wine Wednesday is the same football as simulation `s` in
    Blacksburg. That is what makes the queue's ranking key -- the increment to
    `E[titles] = sum_l P_l(title)` -- a measurement rather than an assumption, and it is
    not something a merge of three independently seeded reports can claim.

    Every failure mode degrades to `None` and the local merge: the module may be absent,
    it may expose a different `action_queue`, it may raise on a league ESPN will not
    serve. It is a sibling being written alongside this one, so its absence is the
    expected case rather than an error.
    """
    seasons = {c.season for c in configs}
    if len(seasons) != 1:
        log.info("leagues span seasons %s; edges.portfolio takes one, ordering here", seasons)
        return None
    try:
        from .edges import portfolio  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - the module is optional by design
        log.debug("no edges.portfolio; the queue orders itself")
        return None
    hook = getattr(portfolio, PORTFOLIO_HOOK, None)
    builder = getattr(portfolio, "build_portfolio", None)
    if not callable(hook) or not callable(builder):
        log.info("edges.portfolio exposes no %s()/build_portfolio(); ordering here", PORTFOLIO_HOOK)
        return None

    season = seasons.pop()
    try:
        book = builder(
            [(c.league_id, ws.my_team_id(c)) for c in configs],
            season,
            seed=ws.seed,
            n_sims=ws.n_sims,
            client=ws.client,
        )
        try:
            queue = hook(book, limit=limit, actionable_only=actionable_only)
        except TypeError:
            # A sibling module under active development: honour its ranking even if the
            # filtering keyword it takes today is not the one it takes tomorrow.
            queue = hook(book, limit=limit)
        actions = [_action_from_item(i, season=season) for i in queue.items]
        failures = [
            {"league": league, "surface": surface, "error": err}
            for league, surface, err in getattr(queue, "failures", ())
        ]
    except Exception as err:
        log.warning("edges.portfolio could not build the queue (%s); ordering here", err)
        return None

    holds = []
    for league in sorted({a.league_name for a in actions}):
        rows = [a for a in actions if a.league_name == league]
        if not any(a.significant and a.actionable for a in rows):
            best = max(rows, key=lambda a: a.delta_title, default=None)
            detail = f" Best found: {best.headline} at {pp(best.delta_title)}." if best else ""
            holds.append(f"{league}: nothing clears its own error and is executable today.{detail}")
    rows = [a.to_dict() for a in actions]
    return _envelope(
        "queue",
        [
            {"ok": True, "league_id": c.league_id, "season": c.season, "name": c.name}
            for c in configs
        ],
        actions=rows,
        holds=holds,
        errors=failures,
        ranked_by=f"edges.portfolio.{PORTFOLIO_HOOK}",
        n_actions=len(actions),
        first_actionable=_first_actionable(rows),
    )


def actions_from(cfg: registry.LeagueConfig, payload: Mapping[str, Any]) -> list[Action]:
    """Every executable thing one league's weekly payload found, priced comparably.

    Only the best candidate per surface. A waiver board's tenth-best claim is not a
    separate decision from its best -- you have one priority to spend -- and listing them
    all would let one league's long tail crowd out another league's real move.
    """
    name = str(payload.get("name") or cfg.name or cfg.league_id)
    team_id = int(payload.get("team_id") or 0)
    out: list[Action] = []

    claims = _get(payload, "waivers", "claims") or []
    free_adds = _get(payload, "waivers", "free_adds") or []
    board = _get(payload, "waivers", "board") or []
    priority = _get(payload, "waivers", "priority")
    uses_faab = bool(_get(payload, "waivers", "uses_faab"))
    # A free add leads over a claim of equal size: same gain, no priority spent, nothing
    # to contest. Falling back to `board` keeps the "nothing is worth doing" row visible.
    actions = free_adds or claims
    top = (actions or board)[:1]
    for row in top:
        free = row.get("on_waivers") is False
        cost = "free" if free else ("FAAB" if uses_faab else "waiver priority")
        spend = (
            ""
            if free
            else (
                f" Spends waiver priority {priority}."
                if priority is not None and not uses_faab
                else ""
            )
        )
        out.append(
            Action(
                league_id=cfg.league_id,
                league_name=name,
                season=cfg.season,
                team_id=team_id,
                surface="waiver",
                headline=("Add " if free else "Claim ")
                + f"{row.get('add', '?')} ({row.get('position', '?')}), "
                f"drop {row.get('drop', '-')}.{spend}",
                delta_title=float(row.get("delta_title") or 0.0),
                stderr=float(row.get("stderr") or 0.0),
                leverage=float(row.get("leverage") or 1.0),
                significant=bool(row.get("significant")),
                verdict=str(row.get("verdict") or "noise"),
                confidence=str(row.get("confidence") or "medium"),
                cost=cost,
                # No hardcoded Wednesday. All three of the user's leagues process on six
                # days at hour 11 and Tuesday is the one day none of them run, so the
                # schedule belongs to the league; the surface's own rationale carries it.
                deadline="now -- first come, no claim" if free else "next waiver run",
                tags=tuple(row.get("tags") or ()),
                rationale=str(row.get("rationale") or ""),
                # A claim can be a real measured gain and still not be worth making: the
                # threshold it has to clear is the continuation value of *holding* the
                # priority for a better week. Significance and executability are two
                # different questions and collapsing them loses the interesting one.
                #
                # A free agent has no such threshold, which is why `actions` and not
                # `claims` decides this. Charging one to him is what turned 24 of 28
                # profitable rows into "hold".
                actionable=bool(actions),
                blockers=()
                if actions
                else ("below the continuation value of holding waiver priority",),
            )
        )

    n_found = _get(payload, "trades", "n_found") or 0
    for row in (_get(payload, "trades", "trades") or [])[:1]:
        gets = ", ".join(p["name"] for p in row.get("receive", [])) or "nothing"
        gives = ", ".join(p["name"] for p in row.get("send", [])) or "nothing"
        partner = ", ".join(p["name"] for p in row.get("partners", [])) or "a rival"
        # The producing surface's warnings plus the one this module owns: the top row of
        # a search is the maximum of n noisy estimates and its point estimate is biased
        # upward. Without this the queue and the headline quoted +1.75pp as if it were
        # a measurement of that trade rather than of the winner of a 35-way argmax.
        trade_caveats = list(row.get("caveats") or caveats_for(row.get("tags") or ()))
        if n_found > 1:
            trade_caveats.append(
                f"best of {n_found} candidates: the point estimate is the maximum of "
                f"{n_found} noisy paired draws and is biased upward"
            )
        out.append(
            Action(
                league_id=cfg.league_id,
                league_name=name,
                season=cfg.season,
                team_id=team_id,
                surface="trade",
                headline=f"Offer {partner}: you get {gets} for {gives}.",
                delta_title=float(row.get("delta_title") or 0.0),
                stderr=float(row.get("stderr") or 0.0),
                leverage=float(row.get("leverage") or 1.0),
                significant=bool(row.get("significant")),
                verdict=str(row.get("verdict") or "noise"),
                confidence=str(row.get("confidence") or "low"),
                cost="negotiation",
                deadline="league trade deadline",
                tags=tuple(row.get("tags") or ()),
                rationale=str(row.get("rationale") or ""),
                caveats=tuple(trade_caveats),
            )
        )

    changes = _get(payload, "lineup", "changes") or []
    if changes:
        rec = _get(payload, "lineup", "recommendation") or {}
        week = _get(payload, "lineup", "week")
        out.append(
            Action(
                league_id=cfg.league_id,
                league_name=name,
                season=cfg.season,
                team_id=team_id,
                surface="lineup",
                headline="Week {}: {}.".format(
                    week, "; ".join(f"start {c['in']} over {c['out']}" for c in changes)
                ),
                delta_title=float(rec.get("delta_title") or 0.0),
                stderr=float(rec.get("stderr") or 0.0),
                leverage=float(_get(payload, "lineup", "leverage") or 1.0),
                significant=bool(rec.get("significant")),
                verdict=str(rec.get("verdict") or "noise"),
                confidence=str(rec.get("confidence") or "medium"),
                cost="free",
                deadline="kickoff",
                tags=tuple(rec.get("tags") or ()),
                rationale=str(rec.get("rationale") or ""),
            )
        )

    stream = _get(payload, "stream", "stream")
    if stream:
        # A hold used to be dropped here entirely, which made `fq queue --local` and the
        # default `fq queue` describe different worlds: the portfolio path puts the three
        # rest-of-season D/ST plans on TOP of the board at +2.0 to +3.8pp and the local
        # merge did not list them at all. Carried instead as not-actionable with the
        # blocker named, which is how the waiver hold on this same board is already
        # treated, and which `rank_actions` sorts below anything executable.
        held = stream.get("action") == "hold"
        gets = ", ".join(p["name"] for p in stream.get("receive", [])) or "-"
        gives = ", ".join(p["name"] for p in stream.get("send", [])) or "-"
        summary = stream.get("plan_summary") or {}
        plan_caveats = list(stream.get("caveats") or caveats_for(stream.get("tags") or ()))
        if held and summary:
            plan_caveats.insert(
                0,
                f"prices {summary.get('n_acquisitions', '?')} acquisition(s) over "
                f"{summary.get('weeks', '?')} weeks; week one's gain over holding is "
                f"{summary.get('week_one_gain_points', 0.0):+.1f} pts",
            )
        out.append(
            Action(
                league_id=cfg.league_id,
                league_name=name,
                season=cfg.season,
                team_id=team_id,
                surface="stream",
                headline=(
                    f"Rest-of-season {stream.get('position', '')} plan "
                    f"(week {_plan_week_one(stream)} action: hold "
                    f"{gives if gives != '-' else 'what you have'})."
                )
                if held
                else f"Stream {stream.get('position', '')}: add {gets}, drop {gives}.",
                delta_title=float(stream.get("delta_title") or 0.0),
                stderr=float(stream.get("stderr") or 0.0),
                leverage=float(stream.get("leverage") or 1.0),
                significant=bool(stream.get("significant")),
                verdict=str(stream.get("verdict") or "noise"),
                confidence=str(stream.get("confidence") or "medium"),
                # Usually free, and the module knows it: `decide/streaming.py` notes
                # that a 14-team league rosters 15 of the 32 defences, so 17 are plain
                # free agents costing nothing at all. Flatly labelling every streaming
                # move "waiver priority" priced a free add as if it spent the scarcest
                # thing on the board.
                cost=_streaming_cost(payload, stream),
                deadline="next waiver run",
                tags=tuple(stream.get("tags") or ()),
                rationale=str(stream.get("rationale") or ""),
                actionable=not held,
                blockers=("no-action-this-week",) if held else (),
                caveats=tuple(plan_caveats),
            )
        )
    return out


def _plan_week_one(stream: Mapping[str, Any]) -> str:
    """The first week of the streaming plan, off the plan itself. Never a constant.

    `fq weekly` is run every week and the queue row said "week 1" in all of them.
    """
    plan = stream.get("plan") or []
    return str(plan[0].get("week")) if plan else "1"


def _best_action(payload: Mapping[str, Any]) -> Action | None:
    """This league's single best action, or None when it found nothing to do."""
    cfg = registry.LeagueConfig(
        league_id=int(payload.get("league_id") or 0),
        season=int(payload.get("season") or 0),
        name=str(payload.get("name") or ""),
    )
    actions = actions_from(cfg, payload)
    if not actions:
        return None
    return rank_actions(actions, limit=1)[0]


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_odds(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    table = Table(
        title=f"{payload['name']} ({payload['league_id']}) -- championship odds, "
        f"{payload['n_sims']:,} sims",
        title_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("team")
    table.add_column("record", justify="right")
    table.add_column("title", justify="right")
    table.add_column("+/-", justify="right")
    table.add_column("vs next", justify="right")
    table.add_column("playoffs", justify="right")
    table.add_column("bye", justify="right")
    table.add_column("E[wins]", justify="right")
    for row in payload["teams"]:
        style = "bold cyan" if row["is_me"] else ""
        record = (
            f"{row['wins']}-{row['losses']}" + (f"-{row['ties']}" if row["ties"] else "")
            if row["wins"] is not None
            else "-"
        )
        sep = row.get("separated_from_next")
        table.add_row(
            str(row["rank"]),
            row["name"] + ("  <- you" if row["is_me"] else ""),
            record,
            pct(row["championship"], 2),
            f"{row.get('championship_stderr', 0.0) * 100:.2f}pp",
            "-" if sep is None else ("yes" if sep else Text("tied", style="dim italic")),
            pct(row["playoffs"]),
            pct(row["bye"]),
            f"{row['expected_wins']:.1f}",
            style=style,
        )
    out.print(table)
    out.print(Text(f"  baseline: {payload['baseline']}", style="dim"))
    # The ranking was the one number in this module published with no error at all, and
    # it is the first thing the weekly report prints. Say how much of it is real.
    if payload.get("ranking_note"):
        out.print(
            Text(
                f"  {payload['ranking_note']}",
                style="dim" if payload.get("n_ranks_separated") else "yellow",
            )
        )
    if payload["corpus_variant"] != payload["corpus_variant_requested"]:
        out.print(
            Text(
                f"  note: no {payload['corpus_variant_requested']!r} corpus for "
                f"{payload['season']}; component stats read from the "
                f"{payload['corpus_variant']!r} capture and re-scored with this league's "
                "own scoring function. Run `fq snapshot` to capture it properly.",
                style="yellow",
            )
        )


def render_leverage(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    this = payload.get("this_week")
    if this is None:
        out.print(Text("  no remaining matchups", style="dim"))
        return
    verdict = "noise" if this["decided"] else "act"
    out.print(
        Text.assemble(
            ("this week  ", "bold"),
            (f"vs {this['opponent']}  ", ""),
            (f"{signed(this['margin'])} +/- {this['sd_diff']:.1f}  ", ""),
            (f"P(win) {pct(this['win_probability'])}  ", ""),
            (f"leverage {this['leverage']:.2f}", _style_for(verdict) or "bold"),
            (
                f"   ({this['points_per_win_pct']:.2f}pp of the game per projected point)",
                "dim",
            ),
        )
    )
    if this["decided"]:
        out.print(
            Text(
                "  This week is effectively decided: a marginal lineup upgrade cannot "
                "change it. Spend the attention elsewhere.",
                style="yellow",
            )
        )
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("wk", justify="right")
    table.add_column("opponent")
    table.add_column("margin", justify="right")
    table.add_column("sd", justify="right")
    table.add_column("P(win)", justify="right")
    table.add_column("leverage", justify="right")
    for row in payload["weeks"]:
        table.add_row(
            str(row["week"]),
            row["opponent"],
            signed(row["margin"]),
            f"{row['sd_diff']:.1f}",
            pct(row["win_probability"]),
            _cell(f"{row['leverage']:.2f}", "noise" if row["decided"] else "act"),
        )
    out.print(table)
    out.print(
        Text(
            f"  mean leverage over the remaining schedule {payload['mean_leverage']:.2f}",
            style="dim",
        )
    )
    # "This week barely matters" is the most valuable thing this section can say and in
    # week 1 it is never true: at 0-0 the projected margin between any two teams is at
    # most ~19 points against an sd_diff of 23-30, so every game is a coin flip by
    # construction. Live, across all 42 remaining matchups in all three leagues, leverage
    # ran 0.706 to 1.000 with a mean of 0.97 and not one row below 0.25. A column that is
    # a constant should say it is a constant rather than look like a ranking.
    values = [r["leverage"] for r in payload["weeks"]]
    if values and min(values) >= DECIDED_LEVERAGE:
        out.print(
            Text(
                f"  No remaining matchup is decided (leverage {min(values):.2f}-{max(values):.2f}, "
                f"floor {DECIDED_LEVERAGE:.2f}): every point lands in every week, so nothing "
                "here separates one week from another yet. This column only becomes "
                "actionable once records and rosters diverge.",
                style="dim",
            )
        )


def render_lineup(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    if not payload["current_known"]:
        out.print(
            Text(
                "  Could not read your currently-set lineup; the deltas below are "
                "against the projected-best lineup, not against what you have set.",
                style="yellow",
            )
        )
    if payload["unpriced_current"]:
        out.print(
            Text(
                f"  WARNING: {len(payload['unpriced_current'])} of your set starters cannot be "
                "legally assigned this week (a bye, or an out player). Your lineup is broken.",
                style="bold red",
            )
        )
    table = Table(title=f"week {payload['week']} lineup", title_style="bold", box=None)
    table.add_column("slot")
    table.add_column("start")
    table.add_column("change")
    changed = {c["slot_id"]: c for c in payload["changes"]}
    for row in payload["recommended"]:
        change = changed.get(row["slot_id"])
        note = (
            Text(f"<- {change['out']} ({signed(change['d_mean'])} pts)", style="bold yellow")
            if change
            else Text("", style="dim")
        )
        table.add_row(row["slot"], row["name"], note)
    out.print(table)
    if not payload["changes"]:
        out.print(Text("  No change: the lineup you have set is the one to start.", style="dim"))
    rec = payload["recommendation"]
    out.print(
        Text.assemble(
            ("  matchup ", "dim"),
            (f"{signed(payload['margin'])} +/- {payload['sd_diff']:.1f}  "),
            (f"z {payload['z']:+.2f}  leverage {payload['leverage']:.2f}   "),
            (
                f"dP(win) {pp(payload['delta_win_prob'])} "
                f"for -{payload['points_sacrifice']:.1f} pts"
                f"  dTitle {pp(rec['delta_title'], 3)} +/-{rec['stderr'] * 100:.3f}pp"
                f"{_marker(rec['verdict'])}",
                _style_for(rec["verdict"]),
            ),
        )
    )
    if payload["guard"]:
        out.print(Text(f"  guard: {payload['guard']}", style="dim"))


def _streaming_cost(payload: Mapping[str, Any], stream: Mapping[str, Any]) -> str:
    """What the streamed add actually costs, read off the waiver board's availability.

    Falls back to "waiver priority" when the board did not report one, which is the
    expensive assumption and the right one to make on no information.
    """
    on_waivers = _get(payload, "waivers", "board") or []
    status = {row.get("add"): row.get("on_waivers", True) for row in on_waivers}
    gets = stream.get("add") or stream.get("gets")
    if isinstance(gets, str) and status.get(gets) is False:
        return "free"
    return "waiver priority"


def _availability_note(payload: Mapping[str, Any]) -> str:
    """"29 on waivers, 782 free" -- the fact that makes the threshold column readable.

    Empty when ESPN was not asked, because the honest reading of a missing answer is that
    every candidate was priced as if it cost a claim, and saying nothing is better than
    printing a count we do not have.
    """
    on_waivers = payload.get("n_on_waivers")
    free = payload.get("n_free_agents_available")
    if on_waivers is None or free is None:
        return "waiver status unread -- every row priced as if it cost a claim"
    return f"{on_waivers} on waivers, {free} free"


def _rankings_line(ranked: Mapping[str, Any]) -> str:
    """One line naming the board a surface was priced against, and that it is unverified."""
    fit = "" if ranked["matches_league_scoring"] else f" ({ranked['scoring']} board; ~1 rank drift)"
    return (
        f"priced against the {ranked['kind']} board, {ranked['n']} players, weight "
        f"{ranked['weight']:.1f}{fit} -- unverified against results; boards are being archived."
    )


def render_waivers(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    if payload["uses_faab"]:
        cost = f"FAAB ${payload['budget']}"
    else:
        # An unreadable waiver order is not "priority None": the threshold below is only
        # as good as this number, so a guess has to be labelled a guess.
        where = "unknown" if payload["priority"] is None else str(payload["priority"])
        flag = "" if payload["priority_known"] else " (ASSUMED -- ESPN would not say)"
        cost = f"waiver priority {where}{flag}"
    out.print(
        Text.assemble(
            ("waivers  ", "bold"),
            (f"{cost}   "),
            (f"claim threshold {payload['threshold'] * 100:.3f}pp   ", "bold"),
            (_availability_note(payload) + "\n  ", "dim"),
            (
                f"title {pct(payload['baseline_title'], 2)} with the wire floor, "
                f"{payload['title_per_point'] * 100:+.4f}pp per ROS point "
                f"(+/-{payload.get('title_per_point_stderr', 0.0) * 100:.4f})",
                "dim",
            ),
        )
    )
    # Every dTitle below is that one rate times that row's paired points gain, so the
    # rate's error is COMMON to the whole board: the differences between rows are far
    # better determined than the levels, and all the levels move together. Re-seeding
    # the live Blacksburg board moved the rate 0.0746 -> 0.0859pp/pt (14%) while keeping
    # the same five players in the same order.
    out.print(
        Text(
            "  every dTitle below is that one rate x that row's points gain, so the rate's "
            "error is common to the board: the ORDER is far better determined than the levels.",
            style="dim",
        )
    )
    table = Table(box=None, pad_edge=False)
    table.add_column("add")
    table.add_column("pos", justify="right")
    table.add_column("drop")
    table.add_column("dPts", justify="right")
    table.add_column("dTitle", justify="right")
    table.add_column("+/-", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("clears?", justify="right")
    ranked = payload.get("rankings")
    if ranked:
        table.add_column("board", justify="right")
    ceilings = any(
        r.get("drop_ceiling") and r["drop_ceiling"] != "-" for r in payload.get("board", [])
    )
    if ceilings:
        table.add_column("drop ceiling", justify="right")
    if not payload["board"]:
        out.print(Text("  Nothing on the wire projects above replacement.", style="dim"))
        return
    fence = 0
    for row in payload["board"]:
        verdict = row["verdict"]
        on_waivers = row.get("on_waivers", True)
        clears = "yes" if row["clears_threshold"] else "no"
        # A `?` is not decoration: this row's distance from the threshold is inside its
        # own error, so which side of the cut it lands on is a coin flip on this draw.
        # It says nothing about a free agent, who has no threshold to be near.
        if on_waivers and not row.get("clears_certain", True):
            clears += "?"
            fence += 1
        if not on_waivers:
            clears = "n/a"
        add = str(row["add"])
        note = row.get("note")
        if note and note != "-":
            add += f"\n  {note}"
        cells = [
            _cell(add, verdict),
            _cell(str(row["position"]), verdict),
            _cell(str(row["drop"]), verdict),
            _cell(signed(row["delta_points"]), verdict),
            _cell(pp(row["delta_title"], 3) + _marker(verdict), verdict),
            _cell(f"{row['stderr'] * 100:.3f}pp", verdict),
            _cell("claim" if on_waivers else "free", verdict),
            _cell(clears, verdict),
        ]
        if ranked:
            board = row.get("board")
            cells.append(_cell("" if not board or board == "-" else board, verdict))
        if ceilings:
            c = row.get("drop_ceiling")
            cells.append(_cell("" if not c or c == "-" else c, verdict))
        table.add_row(*cells)
    out.print(table)
    if ceilings:
        out.print(
            Text(
                "  'drop ceiling' is what that player costs as lineups are actually set, "
                "against what he would have been worth to someone who knew which weeks to "
                "start him. The second number is NOT charged: a lineup set on projections "
                "already captures 0.89 of the ceiling and real managers capture 0.78, so "
                "nobody measured has beaten the projection. It is there so the cut is your "
                "call and not the model's.",
                style="dim",
            )
        )
    stream = payload.get("stream_advantage") or []
    big = [r for r in stream if r["gap"] > 0]
    if big:
        best = max(big, key=lambda r: r["gap"])
        out.print(
            Text(
                "  streaming vs holding, per single-body slot: "
                + ", ".join(f"{r['position']} {r['gap']:+.0f}" for r in big)
                + f" pts over the rest of the season. The gap is non-negative by "
                f"construction, so read the RATIO: {best['position']} is the seat worth "
                f"streaming. Neither number pays for the weekly transaction -- `fq stream` "
                f"plans that properly.",
                style="dim",
            )
        )
    if ranked:
        out.print(Text("  " + _rankings_line(ranked), style="dim"))
    if fence:
        out.print(
            Text(
                f"  {fence} row(s) marked '?' sit closer to the threshold than their own "
                "error, so which side of the cut they land on is a coin flip on this draw "
                "-- and the threshold itself carries an error this does not include. The "
                "claim count below is that uncertain at its bottom end.",
                style="yellow",
            )
        )
    free_adds = payload.get("free_adds") or []
    if free_adds:
        best = free_adds[0]
        # ALTERNATIVES, not a shopping list. Every one of these was priced on its own
        # against today's roster and most of them drop the same player, so they are
        # mutually exclusive and their gains are emphatically not additive. Printing
        # "add 23 free agents" would be the same additivity error the claim waterfall
        # already warns about, in a place where nothing stops the reader acting on it.
        alternatives = len(free_adds) - 1
        drops = {str(row.get("drop", "-")) for row in free_adds}
        tail = ""
        if alternatives:
            tail = (
                f" {alternatives} other free add(s) also help, but they are ALTERNATIVES "
                f"to this one, not additions"
                + (
                    " -- they all drop the same player."
                    if len(drops) <= 1
                    else f": they drop {len(drops)} different players between them, and each "
                    "was priced on its own against today's roster. Make one, then re-run."
                )
            )
        out.print(
            Text(
                f"  Add {best['add']} ({best['position']}) NOW for "
                f"{pp(best['delta_title'], 3)}, dropping {best.get('drop', '-')} -- he is a "
                f"plain free agent, so no claim, no priority, first come.{tail}",
                style="green",
            )
        )
    if payload["any_claim"]:
        claims = payload["claims"]
        order = ", ".join(str(row["add"]) for row in claims[:5])
        if len(claims) > 5:
            order += f", and {len(claims) - 5} more"
        out.print(
            Text(
                f"  Submit {len(claims)} claim(s), in this order: {order}. "
                f"{payload['waterfall_note']}",
                style="green",
            )
        )
    elif not free_adds:
        out.print(
            Text(
                "  No claim clears the cost of spending priority. Hold the queue position.",
                style="dim",
            )
        )
    else:
        out.print(
            Text(
                "  Nothing that would COST a claim is worth one; hold the queue position.",
                style="dim",
            )
        )


def render_trades(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    if not payload["trades"]:
        out.print(
            Text(
                f"  No Pareto-improving trade found ({payload['n_found']} survived the search).",
                style="dim",
            )
        )
        return
    ranked = payload.get("rankings")
    table = Table(box=None, pad_edge=False)
    table.add_column("partner")
    table.add_column("you get")
    table.add_column("you give")
    table.add_column("dPts", justify="right")
    if ranked:
        # The arbitrage, side by side: `dPts` is mine by the analyst board, `spread` is
        # how much better the other side reads the deal by the projections on THEIR
        # screen than by ours. Positive is the case worth having.
        table.add_column("spread", justify="right")
    table.add_column("dTitle", justify="right")
    table.add_column("+/-", justify="right")
    table.add_column("z", justify="right")
    for row in payload["trades"]:
        verdict = row["verdict"]
        what = ", ".join(p["name"] for p in row["receive"])
        for note in row.get("caveats", []):
            what += f"\n  caveat: {note}"
        for who, note in (row.get("notes") or {}).items():
            what += f"\n  {who}: {note}"
        if row.get("same_return"):
            what += "\n  same return as a row above, routed through someone else"
        elif row.get("routes"):
            n = row["routes"]
            plural = "s" if n > 1 else ""
            what += f"\n  {n} other route{plural} to this return; this needs fewest to agree"
        cells = [
            _cell(", ".join(p["name"] for p in row["partners"]), verdict),
            _cell(what, verdict),
            _cell(", ".join(p["name"] for p in row["send"]), verdict),
            _cell(signed(row["delta_points"]), verdict),
        ]
        if ranked:
            spread = row.get("spread")
            cells.append(_cell("" if spread is None else signed(spread), verdict))
        cells.extend(
            [
                _cell(pp(row["delta_title"]) + _marker(verdict), verdict),
                _cell(f"{row['stderr'] * 100:.2f}pp", verdict),
                _cell("-" if row["z"] is None else f"{row['z']:.1f}", verdict),
            ]
        )
        table.add_row(*cells)
    out.print(table)
    if ranked:
        out.print(Text("  " + _rankings_line(ranked), style="dim"))
        out.print(
            Text(
                "  dPts is yours by the analyst board; spread is how much better the other "
                "side reads it by the projections on their own screen. dTitle is priced on "
                "the board too, so compare its deltas across rows, not its level to `fq odds`.",
                style="dim",
            )
        )
    out.print(Text(f"  significance: {payload['significance_test']}", style="dim"))
    if payload.get("selection_note"):
        out.print(Text(f"  {payload['selection_note']}", style="yellow"))


def render_stream(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    row = payload["stream"]
    verdict = row["verdict"]
    out.print(
        Text.assemble(
            (f"streaming {row['position']}  ", "bold"),
            (f"this week: {row['action']}   "),
            (
                f"plan worth {pp(row['delta_title'])} +/-{row['stderr'] * 100:.2f}pp"
                f"{_marker(verdict)}",
                _style_for(verdict),
            ),
        )
    )
    if "no-action-this-week" in row["tags"]:
        out.print(
            Text(
                "  Nothing to execute today: that number prices the whole rest-of-season "
                "plan, not this week's move.",
                style="dim",
            )
        )
    # The single largest number this module prints is a plan value, and three things
    # about it change what it means. All three are arithmetic on the plan table.
    summary = row.get("plan_summary") or {}
    if summary:
        out.print(
            Text(
                f"  What that prices: {summary['n_acquisitions']} acquisition(s) over "
                f"{summary['weeks']} weeks, worth {summary['total_gain_points']:+.1f} model "
                f"points against holding, of which {summary['week_one_gain_points']:+.1f} is "
                f"week {row['plan'][0]['week'] if row['plan'] else '?'} -- the part you can "
                "act on today. You collect the rest only by re-solving every week and "
                "winning every one of those adds.",
                style="yellow" if summary["week_one_gain_points"] <= 0 else "dim",
            )
        )
        if summary["unpriced_share"] > 0.25:
            out.print(
                Text(
                    f"  {summary['unpriced_share'] * 100:.0f}% of it "
                    f"({summary['unpriced_gain_points']:+.1f} pts) is in weeks with no closing "
                    "line yet, priced off the projection-only fit. That share is the least "
                    "trustworthy part of the number and it will move as the market opens.",
                    style="yellow",
                )
            )
    for note in row.get("caveats") or []:
        if "closing line" not in note:  # already said, in numbers, just above
            out.print(Text(f"  caveat: {note}", style="yellow"))
    if row["plan"]:
        shown, total = summary.get("weeks_shown", 0), summary.get("weeks", 0)
        if total > shown:
            out.print(
                Text(
                    f"  showing the first {shown} of {total} planned weeks (--weeks to see more)",
                    style="dim",
                )
            )
        table = Table(box=None, pad_edge=False)
        table.add_column("wk", justify="right")
        table.add_column("start")
        table.add_column("instead of")
        table.add_column("gain", justify="right")
        table.add_column("line?", justify="right")
        for week in row["plan"]:
            table.add_row(
                str(week["week"]),
                week["start"],
                week["hold"],
                signed(week["gain"], 2),
                "yes" if week["opponent_priced"] else "no",
            )
        out.print(table)


def render_weekly(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    if not payload.get("ok", True):
        out.print(
            Text(
                f"{payload.get('name') or payload['league_id']} ({payload['league_id']}): "
                f"{payload['error']}",
                style="bold red",
            )
        )
        return
    out.rule(f"[bold]{payload['name']}[/bold] ({payload['league_id']})  week {payload['week']}")
    out.print(Text(payload["headline"], style="bold"))
    out.print()
    renderers: dict[str, Callable[[Mapping[str, Any], Console], None]] = {
        "odds": render_odds,
        "leverage": render_leverage,
        "lineup": render_lineup,
        "waivers": render_waivers,
        "trades": render_trades,
        "stream": render_stream,
    }
    for name in WEEKLY_SECTIONS:
        section = payload.get(name)
        if not isinstance(section, Mapping):
            continue
        if not section.get("ok", True):
            out.print(Text(f"{name}: {section['error']}", style="red"))
            out.print()
            continue
        renderers[name](section, out)
        out.print()
    out.print(_legend())


def render_queue(payload: Mapping[str, Any], out: Console | None = None) -> None:
    out = out or console
    actions = payload["actions"]
    if not actions:
        out.print(
            Text(
                "Nothing to do in any league. Every surface either found no move or found "
                "one it cannot distinguish from zero.",
                style="bold",
            )
        )
    else:
        table = Table(title="action queue -- all leagues, one unit", title_style="bold")
        table.add_column("#", justify="right")
        table.add_column("league")
        table.add_column("do")
        table.add_column("dTitle", justify="right")
        table.add_column("+/-", justify="right")
        table.add_column("cost")
        table.add_column("by")
        table.add_column("now?", justify="right")
        for i, row in enumerate(actions, start=1):
            verdict = row["verdict"]
            blockers = row.get("blockers") or []
            what = f"[{row['surface']}] {row['headline']}"
            if blockers:
                what += f"\n  -> {', '.join(blockers)}"
            for note in row.get("caveats") or []:
                what += f"\n  caveat: {note}"
            table.add_row(
                str(i),
                _cell(row["league_name"], verdict),
                _cell(what, verdict),
                _cell(pp(row["delta_title"]) + _marker(verdict), verdict),
                _cell(f"{row['stderr'] * 100:.2f}pp", verdict),
                _cell(row["cost"], verdict),
                _cell(row["deadline"].split("(")[0].strip(), verdict),
                _cell("yes" if row.get("actionable", True) else "no", verdict),
            )
        out.print(table)
        first = payload.get("first_actionable")
        if first is None:
            out.print(
                Text(
                    "  Nothing above is both established and executable today. Every row is "
                    "either inside its own error or a plan whose first week is a hold.",
                    style="bold yellow",
                )
            )
        else:
            row = actions[first]
            out.print(
                Text(
                    f"  Do now: #{first + 1} in {row['league_name']} -- {row['headline']} "
                    f"({pp(row['delta_title'])}, {row['cost']}, by {row['deadline']})",
                    style="bold green",
                )
            )
            for note in row.get("caveats") or []:
                out.print(Text(f"    caveat: {note}", style="yellow"))
    for note in payload.get("holds", []):
        out.print(Text(f"  {note}", style="dim"))
    for err in payload.get("errors", []):
        # Two shapes, because two producers: a league this module could not build at all,
        # and one surface `edges.portfolio` lost inside a league it did build.
        where = err.get("league_id") or err.get("league") or "?"
        surface = f" [{err['surface']}]" if err.get("surface") else ""
        out.print(Text(f"  {where}{surface}: {err['error']}", style="red"))
    out.print(Text(f"  ordered by {payload['ranked_by']}", style="dim"))
    out.print(_legend())


# --------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------

app = typer.Typer(help="Reporting: what to do this week, per league and across them.")

LeagueOpt = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="League id or registry name; repeatable. Default: all."),
]
SeasonOpt = Annotated[int | None, typer.Option("--season", help="Override the registry season.")]
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Path to leagues.toml.")]
RankingsOpt = Annotated[
    bool,
    typer.Option(
        "--no-rankings",
        help="Price on ESPN's projections alone, ignoring any analyst board in "
        "data/manual/etr/. The default reads the board when one is there.",
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit the payload as JSON and nothing else.")]
SimsOpt = Annotated[int, typer.Option("--sims", help="Simulations per league.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Common-random-numbers seed.")]
LimitOpt = Annotated[int, typer.Option("--limit", help="Rows to show.")]


def _envelope(command: str, leagues: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "leagues": leagues,
        **extra,
    }


def _emit(payload: Mapping[str, Any], as_json: bool, render: Callable[[], None]) -> None:
    """JSON or a rendering, never both, and never a half-printed table before the JSON."""
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        render()


def _fail(message: str) -> None:
    """A usage error, on stderr so `--json | jq` still gets clean stdout or nothing."""
    Console(stderr=True).print(Text(message, style="bold red"))
    raise typer.Exit(2)


def _run(
    command: str,
    leagues: Sequence[str] | None,
    season: int | None,
    config: Path | None,
    sims: int,
    seed: int,
    as_json: bool,
    build: Callable[[Workspace, registry.LeagueConfig], dict[str, Any]],
    *,
    rankings: bool = True,
) -> dict[str, Any]:
    """Resolve the leagues, run `build` on each, and never let one failure kill the rest.

    The failure path is the point: a league whose credentials expired, whose objective is
    unimplemented, or which ESPN is simply not serving today comes back as one `error`
    row. The other two leagues still print.
    """
    try:
        reg = load_registry(config)
        chosen = select_leagues(reg, leagues, season=season)
    except ReportError as err:
        _fail(str(err))
        raise  # unreachable; `_fail` exits. Kept so the type checker sees no fall-through.
    rows: list[dict[str, Any]] = []
    with Workspace(
        season=season, n_sims=sims, seed=seed, rankings_kind="silva" if rankings else None
    ) as ws:
        for cfg in chosen:
            if not as_json:
                console.print(Text(f"... {cfg.name or cfg.league_id}", style="dim"))
            try:
                rows.append({"ok": True, **build(ws, cfg)})
            except Exception as err:
                log.info("league %s failed", cfg.league_id, exc_info=True)
                rows.append(
                    {
                        "ok": False,
                        "league_id": cfg.league_id,
                        "season": cfg.season,
                        "name": cfg.name,
                        "error": f"{type(err).__name__}: {err}",
                    }
                )
    if rows and not any(r["ok"] for r in rows):
        log.warning("every league failed")
    return _envelope(command, rows)


def _render_each(
    payload: Mapping[str, Any], render: Callable[[Mapping[str, Any], Console], None]
) -> None:
    for row in payload["leagues"]:
        if not row.get("ok", True):
            console.print(
                Text(f"{row.get('name') or row['league_id']}: {row['error']}", style="bold red")
            )
            continue
        console.rule(f"[bold]{row.get('name', '')}[/bold] ({row['league_id']})")
        render(row, console)
    console.print(_legend())


def _sections(skip: Sequence[str] | None) -> tuple[str, ...]:
    """The weekly sections to run, refusing a `--skip` the user misspelled.

    Silently ignoring an unknown section is how a user ends up believing he skipped the
    eight-second trade search when he did not.
    """
    skipped = {s.strip().lower() for s in (skip or [])}
    unknown = sorted(skipped - set(WEEKLY_SECTIONS))
    if unknown:
        _fail(f"unknown section(s) {unknown}; expected one of {', '.join(WEEKLY_SECTIONS)}")
    return tuple(s for s in WEEKLY_SECTIONS if s not in skipped)


@app.command("odds")
def odds_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    as_json: JsonOpt = False,
) -> None:
    """Championship table for one league or all of them."""
    payload = _run("odds", league, season, config, sims, seed, as_json, odds_payload)
    _emit(payload, as_json, lambda: _render_each(payload, render_odds))


@app.command("weekly")
def weekly_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    limit: LimitOpt = 6,
    skip: Annotated[
        list[str] | None,
        typer.Option("--skip", help=f"Sections to skip: {', '.join(WEEKLY_SECTIONS)}."),
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """The whole weekly picture: odds, leverage, lineup, waivers, trades, streaming."""
    sections = _sections(skip)
    payload = _run(
        "weekly",
        league,
        season,
        config,
        sims,
        seed,
        as_json,
        lambda ws, cfg: weekly_payload(ws, cfg, sections=sections, limit=limit),
    )

    def _render() -> None:
        for row in payload["leagues"]:
            render_weekly(row, console)

    _emit(payload, as_json, _render)


@app.command("waivers")
def waivers_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    limit: LimitOpt = 10,
    no_rankings: RankingsOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """The waiver board and the priority threshold a claim has to clear."""
    payload = _run(
        "waivers",
        league,
        season,
        config,
        sims,
        seed,
        as_json,
        lambda ws, cfg: waivers_payload(ws, cfg, limit=limit),
        rankings=not no_rankings,
    )
    _emit(payload, as_json, lambda: _render_each(payload, render_waivers))


@app.command("trades")
def trades_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    limit: LimitOpt = 5,
    min_gain: Annotated[
        float,
        typer.Option(
            "--min-gain",
            help="Playoff-weighted points every counterparty must gain. 0 is the strict "
            "Pareto gate, which passes legs no human accepts; raise it for a shorter, "
            "more negotiable list.",
        ),
    ] = 0.0,
    no_rankings: RankingsOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Search for trades that improve every side, ranked by your title probability."""
    payload = _run(
        "trades",
        league,
        season,
        config,
        sims,
        seed,
        as_json,
        lambda ws, cfg: trades_payload(ws, cfg, limit=limit, min_gain=min_gain),
        rankings=not no_rankings,
    )
    _emit(payload, as_json, lambda: _render_each(payload, render_trades))


@app.command("lineup")
def lineup_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    week: Annotated[int | None, typer.Option("--week", help="Default: the next unplayed.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Start/sit, measured against the lineup you actually have set."""
    payload = _run(
        "lineup",
        league,
        season,
        config,
        sims,
        seed,
        as_json,
        lambda ws, cfg: lineup_payload(ws, cfg, week=week),
    )
    _emit(payload, as_json, lambda: _render_each(payload, render_lineup))


@app.command("stream")
def stream_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    position: Annotated[
        str, typer.Option("--position", help="Position to stream: DST, QB, K, TE.")
    ] = "DST",
    weeks: Annotated[int, typer.Option("--weeks", help="Plan weeks to show.")] = 8,
    as_json: JsonOpt = False,
) -> None:
    """The rest-of-season streaming plan, and what to do about it today."""
    wanted = position.strip().upper().replace("/", "")
    ids = {v.replace("/", ""): k for k, v in POSITION_ABBREV.items()}
    if wanted not in ids:
        _fail(f"unknown position {position!r}; expected one of {', '.join(sorted(ids))}")
    payload = _run(
        "stream",
        league,
        season,
        config,
        sims,
        seed,
        as_json,
        lambda ws, cfg: stream_payload(ws, cfg, position_id=ids[wanted], plan_weeks=weeks),
    )
    _emit(payload, as_json, lambda: _render_each(payload, render_stream))


@app.command("queue")
def queue_command(
    league: LeagueOpt = None,
    season: SeasonOpt = None,
    config: ConfigOpt = None,
    sims: SimsOpt = 4000,
    seed: SeedOpt = 1,
    limit: LimitOpt = 20,
    skip: Annotated[
        list[str] | None,
        typer.Option("--skip", help=f"Sections to skip: {', '.join(WEEKLY_SECTIONS)}."),
    ] = None,
    actionable: Annotated[
        bool,
        typer.Option(
            "--actionable",
            help="Drop rows with nothing to execute today (holds, suppressed plans).",
        ),
    ] = False,
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Merge this module's own per-league reports instead of using "
            "edges.portfolio's shared-season portfolio.",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Every league's best move in one ranked list. The reason delta_title is the unit."""
    sections = _sections(skip)
    payload: dict[str, Any] | None = None
    if not local:
        try:
            reg = load_registry(config)
            chosen = select_leagues(reg, league, season=season)
        except ReportError as err:
            _fail(str(err))
            raise
        if not as_json:
            names = ", ".join(c.name or str(c.league_id) for c in chosen)
            console.print(Text(f"... building a shared-season portfolio over {names}", style="dim"))
        with Workspace(season=season, n_sims=sims, seed=seed) as ws:
            payload = portfolio_queue(ws, chosen, limit=limit, actionable_only=actionable)
    if payload is None:
        inner = _run(
            "queue",
            league,
            season,
            config,
            sims,
            seed,
            as_json,
            lambda ws, cfg: weekly_payload(ws, cfg, sections=sections),
        )
        payload = queue_payload(inner, limit=limit)
    _emit(payload, as_json, lambda: render_queue(payload, console))


def queue_payload(weekly: Mapping[str, Any], *, limit: int = 20) -> dict[str, Any]:
    """Fold every league's weekly payload into one ranked queue, holds stated explicitly.

    A league with nothing to do is not dropped: it gets a line in `holds` saying so in
    its own numbers. A queue that only ever lists actions teaches the user that an empty
    queue means the tool broke.
    """
    actions: list[Action] = []
    holds: list[str] = []
    errors: list[dict[str, Any]] = []
    for row in weekly["leagues"]:
        if not row.get("ok", True):
            errors.append({"league_id": row.get("league_id"), "error": row.get("error")})
            continue
        cfg = registry.LeagueConfig(
            league_id=int(row["league_id"]),
            season=int(row.get("season") or 0),
            name=str(row.get("name") or ""),
        )
        found = actions_from(cfg, row)
        actions.extend(found)
        if not any(a.significant and a.actionable for a in found):
            holds.append(f"{row.get('name')}: {row.get('headline', 'nothing to do')}")
    ordered = rank_actions(actions, limit=limit)
    rows = [a.to_dict() for a in ordered]
    return _envelope(
        "queue",
        list(weekly["leagues"]),
        actions=rows,
        holds=holds,
        errors=errors,
        ranked_by="report.rank_actions",
        n_actions=len(actions),
        first_actionable=_first_actionable(rows),
    )


def mount(parent: typer.Typer) -> typer.Typer:
    """Attach every command here to `parent` as a top-level command.

    One line in `cli.py` -- `from .report import mount; mount(app)` -- and the user gets
    `fq odds` rather than `fq report odds`, which is the whole difference between a tool
    somebody types and one somebody reads the help for.
    """
    for command in app.registered_commands:
        if command not in parent.registered_commands:
            parent.registered_commands.append(command)
    return parent


__all__ = [
    "DECIDED_LEVERAGE",
    "PORTFOLIO_HOOK",
    "SCHEMA_VERSION",
    "WEEKLY_SECTIONS",
    "Action",
    "ReportError",
    "Verdict",
    "Workspace",
    "actions_from",
    "app",
    "caveats_for",
    "champ_stderr",
    "headline",
    "leverage_payload",
    "lineup_payload",
    "load_registry",
    "mount",
    "odds_payload",
    "pct",
    "portfolio_queue",
    "pp",
    "queue_payload",
    "rank_actions",
    "rec_payload",
    "render_leverage",
    "render_lineup",
    "render_odds",
    "render_queue",
    "render_stream",
    "render_trades",
    "render_waivers",
    "render_weekly",
    "select_leagues",
    "signed",
    "slot_label",
    "stream_payload",
    "tag_value",
    "trades_payload",
    "verdict_for",
    "waivers_payload",
    "weekly_payload",
]


if __name__ == "__main__":  # pragma: no cover - `fq` mounts this into cli.py
    app()
