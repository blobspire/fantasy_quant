"""Combining projection sources, at the component level, with equal weights.

Four decisions define this module, and each of them is a place where the obvious
choice is the wrong one.

**Combine components, not points.** A points-level consensus is a consensus *in
one league's scoring*. Ours differ -- 14-team full PPR, two 12-team half PPR --
so a points ensemble would have to be rebuilt per league from sources that mostly
do not publish per-league points anyway. Combining `rec`, `rec_yd`, `rush_att`
and friends and applying `LeagueContext.scorer` afterwards means one ensemble
serves all three, and the half-PPR/full-PPR difference falls out exactly rather
than approximately.

**Equal weights.** Accuracy-weighting is the thing everyone reaches for and it
does not work: over twelve seasons, equal weighting beat accuracy-weighting in
64% of head-to-heads, because source accuracy does not persist from year to year
and a weight fitted on last season is fitted on noise. Weights are supported --
`combine(..., weights=...)` -- for the case where you have a genuine reason, and
the default is equal because the burden of proof is on the exception.

**Hodges-Lehmann location, not the mean and not the median.** The HL estimator is
the median of the Walsh averages: every pairwise mean plus the raw values. With
three to five sources that is the right trade -- about 95% efficiency relative to
the mean under normality (the mean's whole advantage is ~5%), against a 29%
breakdown point, so one source blowing up a player moves the consensus by very
little. A plain median throws away most of the information in four numbers; a
mean hands one bad feed the whole line. Verified against scipy: the Wilcoxon
signed-rank statistic of `values - hodges_lehmann(values)` sits exactly on its
null expectation n(n+1)/4, which is the estimator's defining property.

**Missing sources renormalize; they never impute.** A source that has no line for
a player, or does not publish a stat at all, contributes nothing to that stat's
Walsh set and the estimator is simply taken over the sources that did. No
mean-filling, no zero-filling, no shrinkage to a prior. The cost is that
coverage becomes a real variable, so `source_count` is carried per player and per
stat and exposed -- a one-source line is not a consensus and downstream must be
able to tell.

Two things fall out and are exported because downstream wants them:

* **Disagreement.** The spread of the sources around the consensus is a genuine
  uncertainty signal, and unlike a fitted sigma it is available before kickoff.
  It is exposed per stat (`StatConsensus.spread`) and, more usefully, in points
  under a specific league's scoring (`EnsembleLine.points_spread`).
* **The derived buckets.** ESPN's raw stats contain restatements of themselves
  ("every 5 receiving yards", yards-per-game, total turnovers). `sources.py`
  drops them on ingest so they cannot be averaged against a different source's
  primitives; `add_derived_stats` rebuilds them from the *combined* primitives
  for the handful of leagues whose scoring reads them. The rules were checked
  against 9,360 real ESPN projection rows: zero mismatches.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..core import ComponentLine, LeagueContext
from .sources import (
    DERIVED_STAT_IDS,
    FUMBLES_LOST,
    PASS_ATT,
    PASS_CMP,
    PASS_INT,
    PASS_YDS,
    REC_YDS,
    RECEPTIONS,
    RUSH_ATT,
    RUSH_YDS,
    SourceLinesLike,
)

log = logging.getLogger(__name__)

#: The name the combined line carries.
ENSEMBLE_SOURCE = "ensemble"


# --------------------------------------------------------------------------------------
# Consensus sources: excluded by construction
# --------------------------------------------------------------------------------------


class ConsensusSourceError(ValueError):
    """A source that is itself a consensus was offered as an ensemble input.

    Not a warning and not a silent drop. FantasyPros ECR and FantasyFootballNerd
    are averages of the same feeds we already average -- including one gives the
    sources inside it two votes, and the double-counting is invisible in the
    output because the line still looks like a projection. The failure has to be
    at the door.
    """


#: Vendor roots that are always consensuses of other people's projections.
#: Matched as substrings of the normalized source name, so "fantasypros_ecr" and
#: "FantasyPros Weekly" are both caught.
CONSENSUS_MARKERS: tuple[str, ...] = ("fantasypros", "fantasyfootballnerd")

#: Exact normalized names that mean the same thing. Kept exact rather than fuzzy:
#: "ffn" as a substring would reject a hypothetical source named "ffngrades".
CONSENSUS_ALIASES: frozenset[str] = frozenset(
    {
        "fpros",
        "ffn",
        "ffnerd",
        "ecr",
        "expertconsensus",
        "expertconsensusrankings",
        "consensus",
        # Our own output. Feeding an ensemble back into an ensemble is the same
        # double-count as FantasyPros, arrived at from the other direction.
        ENSEMBLE_SOURCE,
    }
)


def normalize_source_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def is_consensus_source(name: str) -> bool:
    """True for a source that is itself an average of other projections."""
    key = normalize_source_name(name)
    if key in CONSENSUS_ALIASES:
        return True
    return any(marker in key for marker in CONSENSUS_MARKERS)


def reject_consensus_sources(names: Iterable[str]) -> None:
    offenders = sorted({n for n in names if is_consensus_source(n)})
    if offenders:
        raise ConsensusSourceError(
            f"{offenders} are consensus products and cannot be ensemble inputs: they "
            "aggregate the same underlying projections we do, so including one gives "
            "those projections a second vote. Use them to sanity-check the output, "
            "never to build it. (Blocked by CONSENSUS_MARKERS / CONSENSUS_ALIASES in "
            "projections/ensemble.py.)"
        )


# --------------------------------------------------------------------------------------
# Hodges-Lehmann
# --------------------------------------------------------------------------------------


def walsh_averages(values: Sequence[float]) -> list[float]:
    """Every pairwise mean, the raw values included: (x_i + x_j)/2 for i <= j.

    n(n+1)/2 of them. The diagonal i == j is what puts the raw values in the set,
    and leaving it out gives a different (and worse) estimator on small samples.
    """
    n = len(values)
    return [(values[i] + values[j]) / 2.0 for i in range(n) for j in range(i, n)]


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Median of `values` under `weights`, matching `numpy.median` when equal.

    The convention that matters is the tie at exactly half the weight: when the
    cumulative weight lands *on* the halfway point, the two neighbouring values
    are averaged. That is what makes the equal-weight case reproduce the ordinary
    even-length median instead of the lower one.

    Scale-invariant in the weights, which is why "renormalize when a source is
    missing" needs no explicit renormalization step: dropping a source simply
    leaves a smaller weight vector whose total is different and whose median is
    unchanged in meaning.
    """
    if not values:
        raise ValueError("weighted_median of an empty sequence")
    if len(values) != len(weights):
        raise ValueError(f"{len(values)} values against {len(weights)} weights")

    # Zero-weight entries are dropped rather than merely contributing nothing to
    # the running total. Left in, they sit in the sorted order and can be picked
    # as the partner of an exact-half tie -- weights [1, 0, 1] over [1, 2, 3]
    # would return 1.5 instead of 2.0, i.e. the excluded value would decide.
    order = [i for i in sorted(range(len(values)), key=lambda i: values[i]) if weights[i] > 0]
    total = math.fsum(weights[i] for i in order)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    half = total / 2.0
    # Tolerance relative to the total weight, with NO absolute floor. Weights here
    # are products of user-supplied floats, so a floor of `max(total, 1.0)` breaks
    # the scale invariance this function advertises: with four sources weighted
    # 1e-6 each the Walsh pair weights are 1e-12, the whole total is 1e-11, and a
    # 1e-12 absolute tolerance fires the exact-half tie on the *first* value --
    # hodges_lehmann([1, 2, 3, 10], [1e-6]*4) returned 2.25, and at 1e-7 it
    # returned 1.25, against the correct 2.75.
    tol = 1e-12 * total
    running = 0.0
    for position, index in enumerate(order):
        running += weights[index]
        if abs(running - half) <= tol and position + 1 < len(order):
            return (values[index] + values[order[position + 1]]) / 2.0
        if running > half:
            return values[index]
    return values[order[-1]]


def hodges_lehmann(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """The Hodges-Lehmann location estimator: median of the Walsh averages.

    ~95% efficient relative to the mean under normality, with a 29% breakdown
    point -- the right default for a handful of sources, where the median is too
    wasteful and the mean too fragile.

    With `weights`, the Walsh average of sources i and j carries weight w_i*w_j
    (so a source's own value carries w_i^2) and the weighted median is taken.
    Equal weights reduce to the ordinary estimator exactly.

    n == 1 returns the value: a single source is not a consensus, and pretending
    otherwise is what `source_count` exists to prevent.
    """
    n = len(values)
    if n == 0:
        raise ValueError("hodges_lehmann of an empty sequence")
    if n == 1:
        return float(values[0])
    if n == 2 and weights is None:
        # (a+b)/2 either way; short-circuited because two sources is the common case.
        return (float(values[0]) + float(values[1])) / 2.0

    pairs = walsh_averages(values)
    if weights is None:
        return _median(pairs)
    if len(weights) != n:
        raise ValueError(f"{n} values against {len(weights)} weights")
    if any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative, got {list(weights)}")
    pair_weights = [weights[i] * weights[j] for i in range(n) for j in range(i, n)]
    return weighted_median(pairs, pair_weights)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


# --------------------------------------------------------------------------------------
# Combined lines
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatConsensus:
    """One component stat, combined, with the votes that produced it."""

    stat_id: str
    value: float
    #: source name -> that source's value. Contributing sources only; a source
    #: that abstained on this stat is simply absent, never present as a zero.
    values: Mapping[str, float]

    @property
    def source_count(self) -> int:
        return len(self.values)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))

    @property
    def spread(self) -> float:
        """Sample SD across the contributing sources; 0.0 below two of them.

        Sample rather than population because these are a sample of the available
        forecasts, and with n == 2 it makes the spread |a-b|/sqrt(2) rather than
        |a-b|/2, which is the honest scale of the disagreement.
        """
        if len(self.values) < 2:
            return 0.0
        xs = list(self.values.values())
        mean = math.fsum(xs) / len(xs)
        var = math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return math.sqrt(var)

    @property
    def span(self) -> float:
        if not self.values:
            return 0.0
        xs = list(self.values.values())
        return max(xs) - min(xs)


@dataclass(frozen=True, slots=True)
class EnsembleLine:
    """The combined projection for one player-week, plus its provenance.

    Deliberately not a `ComponentLine` subclass. `ComponentLine` is the contract
    for "one source's projection" and stays exactly that; this is the consensus
    over several of them, and it carries the things a consensus has and a single
    source does not -- who voted, how much they disagreed, and how many of them
    there were. `to_component_line` hands back the plain contract type for
    anything that only wants the numbers.
    """

    player_id: int
    season: int
    week: int
    position_id: int
    components: Mapping[str, StatConsensus]
    #: source name -> that source's full stat line, kept so disagreement can be
    #: measured in points under a specific league rather than in raw components.
    per_source: Mapping[str, Mapping[str, float]]
    games: float | None = None
    name: str = ""

    @property
    def stats(self) -> dict[str, float]:
        return {stat_id: c.value for stat_id, c in self.components.items()}

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.per_source))

    @property
    def source_count(self) -> int:
        """Sources with a line for this player. Coverage is informative: a
        one-source player is a projection, not a consensus."""
        return len(self.per_source)

    @property
    def stat_counts(self) -> dict[str, int]:
        return {stat_id: c.source_count for stat_id, c in self.components.items()}

    def to_component_line(self) -> ComponentLine:
        return ComponentLine(
            player_id=self.player_id,
            season=self.season,
            week=self.week,
            source=ENSEMBLE_SOURCE,
            stats=self.stats,
            games=self.games,
        )

    def points(
        self,
        scorer: Callable[[Mapping[str, float], int], float],
        position_id: int | None = None,
        *,
        derived: bool = False,
    ) -> float:
        """Score the consensus under one league's rules.

        `derived=True` rebuilds ESPN's self-restating statIds from the combined
        primitives first. Off by default because none of our leagues price one;
        `Ensemble.score` turns it on by itself when the league does.
        """
        stats = add_derived_stats(self.stats) if derived else self.stats
        return scorer(stats, position_id if position_id is not None else self.position_id)

    def points_by_source(
        self,
        scorer: Callable[[Mapping[str, float], int], float],
        position_id: int | None = None,
        *,
        derived: bool = False,
    ) -> dict[str, float]:
        """Each source's own line, scored in this league. The audit trail."""
        pos = position_id if position_id is not None else self.position_id
        return {
            name: scorer(add_derived_stats(stats) if derived else stats, pos)
            for name, stats in self.per_source.items()
        }

    def points_spread(
        self,
        scorer: Callable[[Mapping[str, float], int], float],
        position_id: int | None = None,
        *,
        derived: bool = False,
    ) -> float:
        """Sample SD of the sources' scored points; 0.0 below two sources.

        This is the disagreement number downstream should actually consume. It is
        league-specific on purpose -- two sources that differ by three receptions
        disagree by 3.0 points in full PPR and 1.5 in half PPR, and pretending
        otherwise would put the same uncertainty on both leagues.

        A sparse source INFLATES it rather than shrinking it: a props line that
        covers only receiving yards scores near zero on everything else, so it
        enters the SD as a low outlier rather than as an abstention. Measured on
        the module's own example -- espn 11.0, sleeper 14.0, props 6.5 -- the
        spread is 3.77 with props and 2.12 without. Prefer `points_spread_over`
        when only the dense sources should count.
        """
        return _sample_sd(self.points_by_source(scorer, position_id, derived=derived).values())

    def points_spread_over(
        self,
        scorer: Callable[[Mapping[str, float], int], float],
        sources: Sequence[str],
        position_id: int | None = None,
        *,
        derived: bool = False,
    ) -> float:
        scored = self.points_by_source(scorer, position_id, derived=derived)
        return _sample_sd([scored[s] for s in sources if s in scored])

    @property
    def mean_stat_spread(self) -> float:
        """Average per-stat spread over the stats more than one source answered."""
        contested = [c.spread for c in self.components.values() if c.source_count > 1]
        return math.fsum(contested) / len(contested) if contested else 0.0


def _sample_sd(values: Iterable[float]) -> float:
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return 0.0
    mean = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


@dataclass(frozen=True, slots=True)
class Ensemble:
    """A whole slate, combined. One of these serves every league."""

    season: int
    week: int
    lines: Mapping[int, EnsembleLine]
    sources: tuple[str, ...]
    weights: Mapping[str, float]

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.lines.values())

    def get(self, player_id: int) -> EnsembleLine | None:
        return self.lines.get(player_id)

    def score(self, league: LeagueContext, *, derived: bool | None = None) -> dict[int, float]:
        """player id -> projected points in this league.

        The reason the whole pipeline is component-level: this call, and only
        this call, is where a league's scoring enters. Run it three times with
        three `LeagueContext`s and the same ensemble answers all three.

        `derived=None` (the default) asks the scorer whether it prices any of the
        statIds `sources.py` drops on ingest, and rebuilds them from the combined
        primitives only if it does. Scoring `stats` blind is the one way this
        pipeline can be quietly wrong by a real number of points: none of our
        three leagues price a derived statId, so the bug would never show up here
        and would show up in full in somebody's every-25-receiving-yards league.
        """
        return {
            pid: line.points(league.scorer, derived=self._use_derived(league, derived))
            for pid, line in self.lines.items()
        }

    def disagreement(
        self, league: LeagueContext, *, derived: bool | None = None
    ) -> dict[int, float]:
        """player id -> SD of the sources' scored points in this league."""
        return {
            pid: line.points_spread(league.scorer, derived=self._use_derived(league, derived))
            for pid, line in self.lines.items()
        }

    def _use_derived(self, league: LeagueContext, derived: bool | None) -> bool:
        if derived is not None:
            return derived
        priced = priced_derived_stats(league.scorer)
        if priced:
            log.warning(
                "league %s prices derived statId(s) %s, which sources.py drops on "
                "ingest; rebuilding them from the combined primitives for this pass. "
                "Reading EnsembleLine.stats directly would under-score by that much.",
                league.league_id,
                sorted(priced, key=int),
            )
        return bool(priced)

    def coverage(self) -> dict[int, int]:
        return {pid: line.source_count for pid, line in self.lines.items()}

    @property
    def unpositioned(self) -> frozenset[int]:
        """Players no source could give a position for. See the warning in `combine`."""
        return frozenset(pid for pid, line in self.lines.items() if not line.position_id)

    def coverage_histogram(self) -> dict[int, int]:
        """source_count -> number of players. Print it weekly; watch it decay."""
        out: dict[int, int] = {}
        for line in self.lines.values():
            out[line.source_count] = out.get(line.source_count, 0) + 1
        return dict(sorted(out.items()))

    def component_lines(self) -> list[ComponentLine]:
        return [line.to_component_line() for line in self.lines.values()]

    def summary(self) -> str:
        histogram = ", ".join(f"{n} source(s): {c}" for n, c in self.coverage_histogram().items())
        return (
            f"season {self.season} week {self.week}: {len(self.lines)} players from "
            f"{list(self.sources)} -- {histogram}"
        )


# --------------------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------------------


def resolve_weights(
    sources: Sequence[str], weights: Mapping[str, float] | None
) -> dict[str, float]:
    """Equal by default. Explicit weights are validated, never guessed at.

    An unknown name is an error rather than a silently ignored key: a typo in a
    weight dict would otherwise leave the source it meant to down-weight at full
    strength, which is the exact failure the weights were written to prevent.
    """
    reject_consensus_sources(sources)
    if weights is None:
        return dict.fromkeys(sources, 1.0)

    reject_consensus_sources(weights)
    unknown = sorted(set(weights) - set(sources))
    if unknown:
        raise ValueError(
            f"weights name sources that are not present: {unknown}; present are {list(sources)}"
        )
    resolved = {s: float(weights.get(s, 1.0)) for s in sources}
    if any(w < 0 for w in resolved.values()):
        raise ValueError(f"weights must be non-negative, got {resolved}")
    if not any(w > 0 for w in resolved.values()):
        raise ValueError("at least one source must carry a positive weight")
    return resolved


def combine(
    sources: Iterable[SourceLinesLike],
    *,
    season: int | None = None,
    week: int | None = None,
    weights: Mapping[str, float] | None = None,
    positions: Mapping[int, int] | None = None,
    names: Mapping[int, str] | None = None,
    min_sources: int = 1,
) -> Ensemble:
    """Combine several sources' lines into one consensus line per player-week.

    `min_sources` filters on coverage after the fact. Leave it at 1: a
    single-source line is still the best estimate available for that player, and
    `source_count` already tells the caller what it is. Raise it only for an
    analysis that genuinely requires agreement.
    """
    collected = list(sources)
    # Positions and names resolve by the SAME precedence, and it has to be stated
    # rather than fallen into: the caller's explicit map wins, then the first
    # source in `sources` order that knows.
    #
    # This used to be `position_map.update(source.positions)`, which meant (a) an
    # explicit `positions=` argument was silently discarded whenever any source
    # also knew the player, and (b) the LAST adapter in the list decided a
    # scoring-critical field. `default_adapters()` puts ESPN first, so Sleeper and
    # ETR were overriding it -- and `pointsOverrides` is keyed on ESPN's
    # `defaultPositionId`, so the wrong answer silently mis-scores a TE-premium or
    # per-position-PPR league with nothing visible downstream. Measured on the live
    # 2026 week-1 slate: ESPN calls Riley Nowakowski (4693370) an RB and Sleeper a
    # TE, and the old order handed the answer to Sleeper.
    position_map: dict[int, int] = {}
    position_votes: dict[int, dict[str, int]] = {}
    name_map: dict[int, str] = dict(names or {})
    for source in collected:
        source_name = getattr(source, "source", type(source).__name__)
        for pid, value in (getattr(source, "positions", {}) or {}).items():
            position_votes.setdefault(pid, {})[source_name] = int(value)
            position_map.setdefault(pid, int(value))
        for pid, value in (getattr(source, "names", {}) or {}).items():
            name_map.setdefault(pid, value)
    position_map.update({int(pid): int(v) for pid, v in (positions or {}).items()})

    disputed = {pid: v for pid, v in position_votes.items() if len(set(v.values())) > 1}
    if disputed:
        log.warning(
            "%d player(s) have a disputed defaultPositionId; the first source that "
            "answered wins (e.g. %s). `pointsOverrides` is keyed on this, so the "
            "loser's position would mis-score a TE-premium or per-position-PPR league.",
            len(disputed),
            {pid: disputed[pid] for pid in sorted(disputed)[:5]},
        )

    lines: list[ComponentLine] = []
    for source in collected:
        lines.extend(source.lines)
    return combine_component_lines(
        lines,
        season=season,
        week=week,
        weights=weights,
        positions=position_map,
        names=name_map,
        min_sources=min_sources,
    )


def combine_component_lines(
    lines: Iterable[ComponentLine],
    *,
    season: int | None = None,
    week: int | None = None,
    weights: Mapping[str, float] | None = None,
    positions: Mapping[int, int] | None = None,
    names: Mapping[int, str] | None = None,
    min_sources: int = 1,
) -> Ensemble:
    """The same, from bare `ComponentLine`s grouped by their own `source` field."""
    grouped: dict[int, dict[str, ComponentLine]] = {}
    seasons: set[int] = set()
    weeks: set[int] = set()
    source_names: list[str] = []

    for line in lines:
        if season is not None and line.season != season:
            continue
        if week is not None and line.week != week:
            continue
        seasons.add(line.season)
        weeks.add(line.week)
        if line.source not in source_names:
            source_names.append(line.source)
        by_source = grouped.setdefault(line.player_id, {})
        if line.source in by_source:
            # Two lines from one source for one player-week would give that source
            # two votes. Keeping the last is arbitrary but at least it is one vote.
            log.warning(
                "duplicate %s line for player %d in %d week %d; keeping the last",
                line.source,
                line.player_id,
                line.season,
                line.week,
            )
        by_source[line.source] = line

    if len(seasons) > 1 or len(weeks) > 1:
        raise ValueError(
            f"combine got a mix of seasons {sorted(seasons)} / weeks {sorted(weeks)}. "
            "Pass `season=`/`week=` to select one -- combining across weeks would "
            "average a bye against a start."
        )

    resolved_weights = resolve_weights(source_names, weights)
    active = [s for s in source_names if resolved_weights[s] > 0.0]
    equal = weights is None or len({resolved_weights[s] for s in active}) <= 1

    out: dict[int, EnsembleLine] = {}
    for player_id, by_source in grouped.items():
        present = {s: line for s, line in by_source.items() if s in active}
        if len(present) < min_sources or not present:
            continue
        out[player_id] = _combine_one(
            player_id=player_id,
            by_source=present,
            weights=resolved_weights,
            equal=equal,
            position_id=(positions or {}).get(player_id, 0),
            name=(names or {}).get(player_id, ""),
        )

    result = Ensemble(
        season=season if season is not None else (seasons.pop() if seasons else 0),
        week=week if week is not None else (weeks.pop() if weeks else 0),
        lines=out,
        sources=tuple(active),
        weights={s: resolved_weights[s] for s in active},
    )
    # A line with no position scores as `defaultPositionId == 0`, which no
    # `pointsOverrides` key matches -- so it silently falls back to the base
    # `points` and a TE-premium or per-position-PPR league under-scores it with no
    # error anywhere. Loud here, because it is invisible downstream.
    if result.unpositioned:
        log.warning(
            "%d of %d ensemble lines carry no position (e.g. %s); they will score "
            "without any pointsOverrides. Supply `positions=` or include a source "
            "that publishes them.",
            len(result.unpositioned),
            len(result.lines),
            sorted(result.unpositioned)[:5],
        )
    return result


def _combine_one(
    *,
    player_id: int,
    by_source: Mapping[str, ComponentLine],
    weights: Mapping[str, float],
    equal: bool,
    position_id: int,
    name: str,
) -> EnsembleLine:
    per_stat: dict[str, dict[str, float]] = {}
    for source, line in by_source.items():
        for stat_id, value in line.stats.items():
            per_stat.setdefault(str(stat_id), {})[source] = float(value)

    components: dict[str, StatConsensus] = {}
    for stat_id, votes in per_stat.items():
        contributors = sorted(votes)
        values = [votes[s] for s in contributors]
        # This is the whole of "renormalize, do not impute": the estimator only
        # ever sees the sources that answered. A source absent from `votes` is not
        # filled with a mean, a zero, or a prior -- it simply does not vote.
        if equal:
            value = hodges_lehmann(values)
        else:
            value = hodges_lehmann(values, [weights[s] for s in contributors])
        components[stat_id] = StatConsensus(stat_id=stat_id, value=value, values=dict(votes))

    games_votes = [line.games for line in by_source.values() if line.games is not None]
    games = hodges_lehmann(games_votes) if games_votes else None

    first = next(iter(by_source.values()))
    return EnsembleLine(
        player_id=player_id,
        season=first.season,
        week=first.week,
        position_id=position_id,
        components=components,
        per_source={s: dict(line.stats) for s, line in by_source.items()},
        games=games,
        name=name,
    )


# --------------------------------------------------------------------------------------
# Derived stats
# --------------------------------------------------------------------------------------

#: statId -> (source statId, divisor). ESPN's "every N yards / attempts" buckets.
#: Each rule verified as exact `floor(source/divisor)` on the 2026 corpus; the
#: rows-checked counts are in `sources.DERIVED_STAT_IDS`. Five of them (13, 14,
#: 32, 52, 54, 55) never appeared in a weekly projection row, so their rule is
#: inferred from the family rather than measured -- flagged here, not hidden.
_FLOOR_BUCKETS: Mapping[str, tuple[str, float]] = {
    "5": (PASS_YDS, 5),
    "6": (PASS_YDS, 10),
    "7": (PASS_YDS, 20),
    "8": (PASS_YDS, 25),
    "9": (PASS_YDS, 50),
    "10": (PASS_YDS, 100),
    "11": (PASS_CMP, 5),
    "12": (PASS_CMP, 10),
    "13": ("2", 5),  # inferred
    "14": ("2", 10),  # inferred
    "27": (RUSH_YDS, 5),
    "28": (RUSH_YDS, 10),
    "29": (RUSH_YDS, 20),
    "30": (RUSH_YDS, 25),
    "31": (RUSH_YDS, 50),
    "32": (RUSH_YDS, 100),  # inferred
    "33": (RUSH_ATT, 5),
    "34": (RUSH_ATT, 10),
    "47": (REC_YDS, 5),
    "48": (REC_YDS, 10),
    "49": (REC_YDS, 20),
    "50": (REC_YDS, 25),
    "51": (REC_YDS, 50),
    "52": (REC_YDS, 100),  # inferred
    "54": (RECEPTIONS, 5),  # inferred
    "55": (RECEPTIONS, 10),  # inferred
}

#: statId -> (numerator, denominator). Zero denominator yields no stat at all,
#: which is what ESPN does: a player with no carries has no yards-per-carry row.
_RATIOS: Mapping[str, tuple[str, str]] = {
    "21": (PASS_CMP, PASS_ATT),
    "39": (RUSH_YDS, RUSH_ATT),
    "60": (REC_YDS, RECEPTIONS),
}

#: statId -> the statIds it restates, summed.
_SUMS: Mapping[str, tuple[str, ...]] = {"73": (PASS_INT, FUMBLES_LOST)}

#: statId -> the statId it copies. Per-game rates equal the total on a one-game week.
_COPIES: Mapping[str, str] = {"22": PASS_YDS, "40": RUSH_YDS, "61": REC_YDS, "41": RECEPTIONS}


def add_derived_stats(stats: Mapping[str, float]) -> dict[str, float]:
    """Rebuild ESPN's self-restating statIds from the combined primitives.

    `sources.py` drops them on ingest so the ensemble never averages receiving
    yards against a *different* source's floor-bucket of receiving yards. A
    league whose scoring reads one of them (a 100-yard-game bonus league, an
    every-25-yards league) needs them back, and because they are exact functions
    of the primitives, recomputing beats carrying them.

    Only for leagues that score them. None of our three do, so this is off by
    default everywhere; call it explicitly when a `LeagueScoring` turns out to
    price statId 47.

    Statement of what is *not* here: the threshold-bonus stats (17/18 "300-399
    yard passing game", 37/38, 56/57, and the 40+/50+ TD bonuses 15/16, 35/36,
    45/46) are probabilities in a projection row, not functions of the means, and
    cannot be derived. They pass through the ensemble as ordinary stats from
    whichever source publishes them.
    """
    out = dict(stats)
    incompletions = out.get(PASS_ATT, 0.0) - out.get(PASS_CMP, 0.0)
    if PASS_ATT in out and PASS_CMP in out:
        out["2"] = incompletions
    for stat_id, source_id in _COPIES.items():
        if source_id in out:
            out[stat_id] = out[source_id]
    for stat_id, (numerator, denominator) in _RATIOS.items():
        den = out.get(denominator)
        if den:
            out[stat_id] = out.get(numerator, 0.0) / den
    for stat_id, parts in _SUMS.items():
        if any(p in out for p in parts):
            out[stat_id] = math.fsum(out.get(p, 0.0) for p in parts)
    for stat_id, (source_id, divisor) in _FLOOR_BUCKETS.items():
        value = out.get(source_id)
        if value is not None:
            out[stat_id] = float(math.floor(value / divisor))
    return out


#: Every defaultPositionId a `pointsOverrides` map can key on for the players we
#: project, plus 0 for the no-position fallback. From espn/constants.py's own list.
_OVERRIDE_POSITION_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 15, 16)


def priced_derived_stats(scorer: object) -> frozenset[str]:
    """The statIds `sources.py` drops on ingest that this scorer actually pays for.

    Duck-typed on `espn.scoring.LeagueScoring` rather than imported, because
    `LeagueContext.scorer` is declared as a bare callable and a test double is
    entitled to be one. A scorer that cannot answer returns the empty set -- the
    status quo rather than a new silent failure -- and `Ensemble.score(...,
    derived=True)` remains available to force the issue.

    `points: 0` is not the same as unscored: an item can carry a positive
    `pointsOverrides` for one position and nothing for the rest, so every position
    id that can key an override is checked rather than the base `points` alone.
    """
    ids = getattr(scorer, "scored_stat_ids", None)
    if ids is None:
        return frozenset()
    candidates = frozenset(str(s) for s in ids) & DERIVED_STAT_IDS
    points_for = getattr(scorer, "points_for", None)
    if points_for is None:
        return candidates
    return frozenset(
        stat_id
        for stat_id in candidates
        if any(points_for(stat_id, position) != 0.0 for position in _OVERRIDE_POSITION_IDS)
    )


def with_derived_stats(line: EnsembleLine) -> ComponentLine:
    """`line.to_component_line()` with the derived statIds restored."""
    base = line.to_component_line()
    return ComponentLine(
        player_id=base.player_id,
        season=base.season,
        week=base.week,
        source=base.source,
        stats=add_derived_stats(base.stats),
        games=base.games,
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedPlayer:
    """One row of a league-specific projection table."""

    player_id: int
    name: str
    position_id: int
    points: float
    spread: float
    source_count: int
    sources: tuple[str, ...] = field(default_factory=tuple)


def rank(
    ensemble: Ensemble, league: LeagueContext, *, limit: int | None = None
) -> list[RankedPlayer]:
    """The ensemble, scored in one league, best first."""
    # Same derived-stat decision as `Ensemble.score`, taken once rather than per row.
    derived = bool(priced_derived_stats(league.scorer))
    rows = [
        RankedPlayer(
            player_id=line.player_id,
            name=line.name,
            position_id=line.position_id,
            points=line.points(league.scorer, derived=derived),
            spread=line.points_spread(league.scorer, derived=derived),
            source_count=line.source_count,
            sources=line.sources,
        )
        for line in ensemble.lines.values()
    ]
    rows.sort(key=lambda r: r.points, reverse=True)
    return rows[:limit] if limit else rows
