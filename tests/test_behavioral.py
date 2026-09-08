"""Behavioral-estimator tests.

Everything here is offline and synthetic, and it is synthetic on purpose. The point of
this module is that most of what it could claim about a real manager is noise, and the
only way to know an estimator recovers a real effect is to plant one of a known size and
demand it back. So each test builds a transaction log by hand with a bias written into
it, runs the estimator, and asserts on the number that comes out.

Four groups, because they fail for different reasons:

*Fitters.* `conditional_logit` and `ols` against closed forms and against the pathology
that actually bites -- separation, which sent the undamped Newton iteration to
coefficients around -1,400 on the real data before the line search went in.

*Estimators.* One planted effect each. The sunk-cost pair is the important one: the data
generator makes draft round predict holding **only through production**, so the naive
coefficient must be large and the controlled one must be ~0. An estimator that cannot
pass that is measuring the draft, not the manager.

*Shrinkage.* The two-transaction manager goes to the mean and the two-hundred-transaction
manager does not, which is the whole reason the empirical-Bayes step exists.

*Refusal.* A trait below its minimum n comes back `estimable=False` with the required n
attached, and never with a number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fantasy_quant.edges.behavioral import (
    MIN_MANAGERS_FOR_SPREAD,
    MIN_N,
    BehavioralError,
    BehavioralPanel,
    PlayerFacts,
    SeasonRecord,
    TraitEstimate,
    activity,
    add_recency,
    behavioral_report,
    conditional_logit,
    draft_holds,
    draft_recency,
    empirical_bayes,
    endowment,
    home_team,
    latency,
    load_player_facts,
    manager_id,
    name_brand,
    ols,
    quiet_hours,
    roster_events,
    sunk_cost,
    trade_receptiveness,
)
from fantasy_quant.espn.league import DraftPick, Transaction, TransactionItem

# --------------------------------------------------------------------------------------
# Synthetic league construction
# --------------------------------------------------------------------------------------

SEASON = 2025
FINAL_WEEK = 14
#: Managers are SWID-shaped because that is what the real identity is.
MANAGERS = [f"{{0000000{i}-0000-0000-0000-00000000000{i}}}" for i in range(1, 9)]

#: A day in milliseconds, and a base timestamp that lands on a Tuesday afternoon ET.
DAY_MS = 86_400_000
BASE_MS = 1_725_000_000_000


def owners(n: int = 8) -> dict[int, str]:
    return {i + 1: manager_id(MANAGERS[i]) for i in range(n)}


def pick(
    team_id: int, overall: int, player_id: int, *, round_id: int | None = None, auto: bool = False
) -> DraftPick:
    return DraftPick(
        id=overall,
        overall_pick_number=overall,
        round_id=round_id if round_id is not None else (overall - 1) // 8 + 1,
        round_pick_number=(overall - 1) % 8 + 1,
        team_id=team_id,
        player_id=player_id,
        lineup_slot_id=20,
        bid_amount=0,
        keeper=False,
        reserved_for_keeper=False,
        nominating_team_id=0,
        auto_drafted=auto,
    )


def item(kind: str, player_id: int, *, frm: int = 0, to: int = 0) -> TransactionItem:
    return TransactionItem(
        type=kind,
        player_id=player_id,
        from_team_id=frm,
        to_team_id=to,
        from_lineup_slot_id=-1,
        to_lineup_slot_id=-1,
        is_keeper=False,
        overall_pick_number=0,
    )


def tx(
    kind: str,
    team_id: int,
    week: int,
    items: list[TransactionItem],
    *,
    stamp: int | None = None,
    status: str = "EXECUTED",
    tx_id: str = "",
) -> Transaction:
    return Transaction(
        id=tx_id or f"{kind}-{team_id}-{week}-{len(items)}-{stamp}",
        type=kind,
        status=status,
        execution_type="EXECUTE",
        scoring_period_id=week,
        team_id=team_id,
        member_id=None,
        bid_amount=0,
        is_pending=False,
        is_league_manager=False,
        proposed_date=BASE_MS + week * 7 * DAY_MS if stamp is None else stamp,
        process_date=None,
        related_transaction_id=None,
        items=tuple(items),
    )


def season_record(
    *,
    picks: list[DraftPick] | None = None,
    transactions: list[Transaction] | None = None,
    season: int = SEASON,
    n_teams: int = 8,
    complete: bool = True,
) -> SeasonRecord:
    return SeasonRecord(
        league_id=1,
        season=season,
        size=n_teams,
        owners=owners(n_teams),
        names={m: f"manager {m[-2]}" for m in owners(n_teams).values()},
        team_names={t: f"Team {t}" for t in owners(n_teams)},
        picks=tuple(picks or []),
        transactions=tuple(transactions or []),
        final_week=FINAL_WEEK,
        complete=complete,
        drafted=True,
        log_available=True,
    )


def panel_of(*records: SeasonRecord, focus: int = SEASON) -> BehavioralPanel:
    return BehavioralPanel(league_id=1, focus_season=focus, seasons=tuple(records))


def facts_from(
    *,
    season_actual: dict[tuple[int, int], float] | None = None,
    week_actual: dict[tuple[int, int, int], float] | None = None,
    season_projection: dict[tuple[int, int], float] | None = None,
    adp: dict[tuple[int, int], float] | None = None,
    position: dict[int, int] | None = None,
    pro_team: dict[tuple[int, int], int] | None = None,
) -> PlayerFacts:
    week_actual = week_actual or {}
    games: dict[tuple[int, int], int] = {}
    for s, _w, p in week_actual:
        games[(s, p)] = games.get((s, p), 0) + 1
    return PlayerFacts(
        season_actual=season_actual or {},
        week_actual=week_actual,
        season_projection=season_projection or {},
        adp=adp or {},
        position=position or {},
        pro_team=pro_team or {},
        games=games,
        seasons=frozenset({SEASON, SEASON - 1}),
    )


# --------------------------------------------------------------------------------------
# Fitters
# --------------------------------------------------------------------------------------


def test_conditional_logit_recovers_a_planted_coefficient() -> None:
    """Simulate choices from a known beta and get it back inside two standard errors."""
    rng = np.random.default_rng(0)
    truth = np.array([1.5, -0.6])
    designs = []
    for _ in range(4000):
        x = rng.normal(size=(8, 2))
        utility = x @ truth + rng.gumbel(size=8)
        chosen = int(utility.argmax())
        order = [chosen, *[i for i in range(8) if i != chosen]]
        designs.append(x[order])
    fit = conditional_logit(designs, ridge=0.0)
    assert fit.converged
    assert np.allclose(fit.beta, truth, atol=0.12)
    assert np.all(np.abs(fit.beta - truth) < 3 * fit.stderr)


def test_conditional_logit_survives_complete_separation() -> None:
    """The chosen alternative is always the best on both regressors: the MLE is at infinity.

    This is not a hypothetical. Real managers add the top free agent at the position most
    weeks, and before the line search went in this exact shape returned coefficients of
    order 1e3 with the wrong sign. The fit must stay finite, converge, and point the
    right way.
    """
    designs = [np.array([[3.0, 3.0], [-1.0, -1.0], [-1.0, -0.5], [-0.5, -1.0]]) for _ in range(60)]
    fit = conditional_logit(designs)
    assert fit.converged
    assert np.all(np.isfinite(fit.beta))
    assert np.max(np.abs(fit.beta)) < 50.0
    assert np.all(fit.beta > 0)


def test_conditional_logit_ridge_scales_with_sample_size() -> None:
    """The same separated design at 20 and 200 choices lands in the same place.

    An absolute ridge would let the coefficients grow tenfold with the data; a per-choice
    prior keeps the penalty per observation constant, which is what makes a 14-add
    manager and a 125-add manager comparable at all.
    """
    design = np.array([[2.0, 0.0], [-1.0, 0.5], [-1.0, -0.5]])
    small = conditional_logit([design] * 20)
    large = conditional_logit([design] * 200)
    assert abs(float(small.beta[0]) - float(large.beta[0])) < 0.25


def test_conditional_logit_rejects_ragged_designs() -> None:
    with pytest.raises(BehavioralError):
        conditional_logit([np.zeros((3, 2)), np.zeros((3, 3))])


def test_ols_matches_the_closed_form() -> None:
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(200), rng.normal(size=200), rng.normal(size=200)])
    beta = np.array([1.0, 2.0, -0.5])
    y = x @ beta + rng.normal(scale=0.4, size=200)
    fit = ols(x, y)
    assert np.allclose(fit.beta, beta, atol=0.1)
    closed = np.linalg.solve(x.T @ x, x.T @ y)
    assert np.allclose(fit.beta, closed)


# --------------------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------------------


def _fake_counts(raw: dict[str, float], n: int) -> dict[str, int]:
    return dict.fromkeys(raw, n)


def test_shrinkage_pulls_a_tiny_sample_to_the_mean_and_leaves_a_large_one_alone() -> None:
    """The headline property: precision decides how much of your own number you keep."""
    # Ten ordinary managers measured on 50 observations each, then the same extreme raw
    # estimate arrived at from 2 observations and from 200. Per-observation sd is 3.0, so
    # the two extremes differ only in how much data is behind them.
    sigma = 3.0
    rng = np.random.default_rng(42)
    raw = {f"m{i}": float(rng.normal(scale=0.5)) for i in range(10)}
    stderr = {f"m{i}": sigma / math.sqrt(50) for i in range(10)}
    counts = _fake_counts(raw, 50)
    raw["noisy"], stderr["noisy"], counts["noisy"] = 3.0, sigma / math.sqrt(2), 2
    raw["solid"], stderr["solid"], counts["solid"] = 3.0, sigma / math.sqrt(200), 200

    table, shrink = empirical_bayes("activity", raw, stderr, counts, required_n=1)
    assert shrink.between_sd > 0

    noisy, solid = table["noisy"], table["solid"]
    assert noisy.raw == solid.raw == 3.0
    # The two-observation manager keeps almost none of his own estimate...
    assert noisy.weight < 0.25
    assert abs(noisy.estimate - shrink.league_mean) < 0.25 * abs(3.0 - shrink.league_mean)
    # ...and the two-hundred-observation one keeps essentially all of it.
    assert solid.weight > 0.9
    assert noisy.weight < 0.25 * solid.weight
    assert abs(solid.estimate - 3.0) < 0.2
    assert solid.notable and not noisy.notable


def test_shrinkage_collapses_when_the_spread_is_pure_error() -> None:
    """Managers drawn from one distribution get pooled, and the report says so."""
    rng = np.random.default_rng(11)
    stderr = {f"m{i}": 1.0 for i in range(12)}
    raw = {f"m{i}": float(rng.normal(scale=1.0)) for i in range(12)}
    table, shrink = empirical_bayes("activity", raw, stderr, _fake_counts(raw, 50), required_n=1)
    assert shrink.collapsed
    assert shrink.between_sd == 0.0
    assert len({round(e.estimate, 9) for e in table.values()}) == 1
    assert not any(e.notable for e in table.values())
    # A collapsed trait must still publish a non-zero interval: the league mean itself is
    # only estimated.
    assert all(e.stderr > 0 for e in table.values())


def test_shrinkage_pools_completely_below_the_manager_floor() -> None:
    raw = {"a": 1.0, "b": -1.0}
    table, shrink = empirical_bayes(
        "endowment", raw, {"a": 0.05, "b": 0.05}, _fake_counts(raw, 50), required_n=1
    )
    assert MIN_MANAGERS_FOR_SPREAD > 2
    assert shrink.collapsed
    assert table["a"].estimate == pytest.approx(table["b"].estimate)


def test_shrinkage_refuses_below_minimum_n_and_says_what_it_needed() -> None:
    raw = {"a": 1.0, "b": 2.0, "c": 3.0}
    counts = {"a": 2, "b": 2, "c": 2}
    table, shrink = empirical_bayes("sunk_cost", raw, dict.fromkeys(raw, 0.1), counts)
    assert shrink.n_managers == 0
    for est in table.values():
        assert not est.estimable
        assert math.isnan(est.estimate) and math.isnan(est.raw)
        assert est.required_n == MIN_N["sunk_cost"]
        assert str(est.required_n) in est.note or "n=" in est.note
        assert "refused" in est.describe()


# --------------------------------------------------------------------------------------
# Sunk cost -- the planted confound
# --------------------------------------------------------------------------------------


def _sunk_cost_league(
    *, sunk_effect: float, seasons: int = 4, seed: int = 5
) -> tuple[BehavioralPanel, PlayerFacts]:
    """Generate holds where draft round predicts holding ONLY through production.

    Early picks are better players (that is what a draft is), better players score more,
    and higher scorers are held longer. Nothing else links round to holding unless
    `sunk_effect` is non-zero. So the naive round coefficient must be large in both cases
    and the controlled one must separate them.
    """
    rng = np.random.default_rng(seed)
    records: list[SeasonRecord] = []
    season_actual: dict[tuple[int, int], float] = {}
    week_actual: dict[tuple[int, int, int], float] = {}
    position: dict[int, int] = {}

    player = 1000
    for s in range(seasons):
        season = SEASON - s
        picks: list[DraftPick] = []
        drops: list[Transaction] = []
        for overall in range(1, 8 * 12 + 1):
            team = (overall - 1) % 8 + 1
            round_id = (overall - 1) // 8 + 1
            player += 1
            position[player] = 2
            # Quality falls with the round, plus real noise: this is the confound.
            quality = 12.0 - 0.7 * round_id + rng.normal(scale=2.0)
            points = max(quality, 0.0) * FINAL_WEEK
            season_actual[(season, player)] = points
            for w in range(1, 5):
                week_actual[(season, w, player)] = max(quality, 0.0)
            picks.append(pick(team, overall, player, round_id=round_id))
            # Holding is driven by production, and by round only through `sunk_effect`.
            held = 3.0 + 0.9 * quality - sunk_effect * round_id + rng.normal(scale=1.0)
            held = int(np.clip(round(held), 1, FINAL_WEEK + 1))
            if held <= FINAL_WEEK:
                drops.append(tx("ROSTER", team, held + 1, [item("DROP", player, frm=team)]))
        records.append(season_record(picks=picks, transactions=drops, season=season))
    return panel_of(*records), facts_from(
        season_actual=season_actual, week_actual=week_actual, position=position
    )


def test_sunk_cost_control_removes_a_planted_confound() -> None:
    """Round predicts holding only through production: naive is large, controlled is ~0."""
    panel, facts = _sunk_cost_league(sunk_effect=0.0)
    table, shrink, fits = sunk_cost(panel, facts)

    naive = float(np.mean([f.naive for f in fits.values()]))
    controlled = float(np.mean([f.controlled for f in fits.values()]))
    assert naive > 1.5, f"the confound should be large, got {naive}"
    assert abs(controlled) < 0.35 * naive, f"the control should remove it, got {controlled}"
    assert abs(controlled) < 0.6

    # And nobody should be flagged as a sunk-cost manager on data with no sunk cost.
    assert not any(e.notable for e in table.values())
    assert shrink.between_sd < 0.6


def test_sunk_cost_recovers_a_planted_effect() -> None:
    """With a real round effect on top of production, the controlled estimate finds it."""
    panel, facts = _sunk_cost_league(sunk_effect=0.5)
    _, _, fits = sunk_cost(panel, facts)
    controlled = float(np.mean([f.controlled for f in fits.values()]))
    naive = float(np.mean([f.naive for f in fits.values()]))
    assert controlled == pytest.approx(0.5 * _round_sd(), rel=0.45), controlled
    assert naive > controlled


def _round_sd() -> float:
    """SD of round id over a 12-round, 8-team draft. Coefficients are per SD of round."""
    rounds = np.repeat(np.arange(1, 13), 8).astype(float)
    return float(rounds.std())


def test_sunk_cost_ignores_the_season_in_progress() -> None:
    """An unfinished season has no hold durations, only censoring, and must not be used."""
    panel, facts = _sunk_cost_league(sunk_effect=0.0, seasons=2)
    finished = panel.seasons
    live = season_record(
        picks=list(finished[-1].picks), transactions=[], season=SEASON + 1, complete=False
    )
    with_live = panel_of(*finished, live, focus=SEASON)
    before, _, _ = sunk_cost(panel, facts)
    after, _, _ = sunk_cost(with_live, facts)
    counted_before = {m: e.n for m, e in before.items()}
    counted_after = {m: e.n for m, e in after.items()}
    assert counted_before == counted_after


def test_draft_holds_censor_at_the_end_of_the_season() -> None:
    picks = [pick(1, 1, 501), pick(1, 2, 502)]
    log = [tx("ROSTER", 1, 6, [item("DROP", 501, frm=1)])]
    holds = {h.player_id: h for h in draft_holds(season_record(picks=picks, transactions=log))}
    assert holds[501].released and holds[501].weeks_held == 5
    assert not holds[502].released and holds[502].weeks_held == FINAL_WEEK


# --------------------------------------------------------------------------------------
# Recency
# --------------------------------------------------------------------------------------


def _recency_league(
    tilts: dict[int, float], *, per_week: int = 2, seed: int = 2
) -> tuple[BehavioralPanel, PlayerFacts]:
    """Free agents whose recent and forward values are independent, and managers who
    choose on one or the other with a planted mix.

    `tilts[team]` is the probability that team takes the pool's best *recent* scorer
    rather than its best *forward* scorer, so the expected rank tilt is monotone in it.
    Each add carries a drop of the team's previous pickup, which is what a real add does
    and which keeps the free-agent pool from draining.
    """
    rng = np.random.default_rng(seed)
    pool = list(range(2000, 2200))
    position = dict.fromkeys(pool, 3)
    week_actual: dict[tuple[int, int, int], float] = {}
    season_actual: dict[tuple[int, int], float] = {(SEASON, p): 100.0 for p in pool}
    # Recent and forward production are deliberately uncorrelated so the two ranks are
    # separately identified; the real data is not like this and the module says so.
    for p in pool:
        for w in range(1, FINAL_WEEK + 1):
            week_actual[(SEASON, w, p)] = float(rng.gamma(2.0, 4.0))

    def forward(p: int, week: int) -> float:
        return sum(week_actual[(SEASON, w, p)] for w in range(week, FINAL_WEEK + 1))

    transactions: list[Transaction] = []
    rostered: set[int] = set()
    held: dict[int, list[int]] = {t: [] for t in tilts}
    stamp = BASE_MS
    for week in range(2, FINAL_WEEK + 1):
        for team, tilt in tilts.items():
            for _ in range(per_week):
                free = [p for p in pool if p not in rostered]
                if rng.random() < tilt:
                    target = max(free, key=lambda p: week_actual[(SEASON, week - 1, p)])
                else:
                    target = max(free, key=lambda p: forward(p, week))
                rostered.add(target)
                items = [item("ADD", target, to=team)]
                if len(held[team]) >= 3:
                    stale = held[team].pop(0)
                    rostered.discard(stale)
                    items.append(item("DROP", stale, frm=team))
                held[team].append(target)
                stamp += 3_600_000
                transactions.append(tx("FREEAGENT", team, week, items, stamp=stamp))
    record = season_record(transactions=transactions)
    return panel_of(record), facts_from(
        season_actual=season_actual, week_actual=week_actual, position=position
    )


def test_add_recency_orders_managers_by_a_planted_tilt() -> None:
    """Teams that chase last week must land above teams that chase forward production."""
    tilts = {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.5, 8: 0.5}
    panel, facts = _recency_league(tilts)
    table, shrink = add_recency(panel, facts)
    by_manager = {t: table[panel.focus.owners[t]] for t in tilts}
    assert all(by_manager[t].estimable for t in tilts)

    chasers = np.mean([by_manager[t].estimate for t in (1, 2)])
    forward = np.mean([by_manager[t].estimate for t in (5, 6)])
    assert chasers > 0.3, chasers
    assert forward < -0.3, forward
    assert chasers - forward > 0.8
    assert shrink.between_sd > 0
    assert by_manager[1].notable and by_manager[5].notable


def test_add_recency_finds_nothing_when_nothing_is_planted() -> None:
    """Every manager on the same rule: the spread must collapse rather than invent ranks."""
    panel, facts = _recency_league(dict.fromkeys(range(1, 9), 0.5), seed=17)
    table, shrink = add_recency(panel, facts)
    assert shrink.collapsed or shrink.signal_to_noise < 1.0
    assert not any(e.notable for e in table.values())


def test_draft_recency_recovers_a_last_season_chaser() -> None:
    """One manager drafts last season's leaderboard; the rest draft the projection."""
    rng = np.random.default_rng(4)
    pool = list(range(3000, 3160))
    position = dict.fromkeys(pool, 2)
    prior = {(SEASON - 1, p): float(rng.uniform(0, 300)) for p in pool}
    projection = {(SEASON, p): float(rng.uniform(0, 300)) for p in pool}

    picks: list[DraftPick] = []
    taken: set[int] = set()
    for overall in range(1, 8 * 14 + 1):
        team = (overall - 1) % 8 + 1
        free = [p for p in pool if p not in taken]
        key = prior if team == 1 else projection
        offset = SEASON - 1 if team == 1 else SEASON
        target = max(free, key=lambda p: key[(offset, p)])
        taken.add(target)
        picks.append(pick(team, overall, target))

    panel = panel_of(season_record(picks=picks))
    facts = facts_from(season_actual=prior, season_projection=projection, position=position)
    table, _ = draft_recency(panel, facts)
    chaser = table[panel.focus.owners[1]]
    others = [table[panel.focus.owners[t]] for t in range(2, 9)]
    assert chaser.estimable and chaser.estimate > 0.4
    assert chaser.estimate > max(o.estimate for o in others) + 0.5
    assert chaser.notable
    assert chaser.detail["logit_share"] > 0.85


def test_recency_reports_the_regressor_correlation_it_had_to_work_with() -> None:
    """When the two measures are the same thing, the caller has to be able to see it."""
    rng = np.random.default_rng(6)
    pool = list(range(4000, 4120))
    position = dict.fromkeys(pool, 4)
    prior = {(SEASON - 1, p): float(rng.uniform(1, 300)) for p in pool}
    # The projection is last season's points plus a whisper: correlation near 1.
    projection = {(SEASON, p): prior[(SEASON - 1, p)] + rng.normal(scale=1.0) for p in pool}
    picks = [pick((i - 1) % 8 + 1, i, pool[i - 1]) for i in range(1, 8 * 12 + 1)]
    panel = panel_of(season_record(picks=picks))
    facts = facts_from(season_actual=prior, season_projection=projection, position=position)
    table, _ = draft_recency(panel, facts)
    assert next(iter(table.values())).detail["regressor_corr"] > 0.95


# --------------------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------------------


def test_endowment_recovers_a_planted_ask_ratio() -> None:
    """A manager who always asks for 1.6x what he offers reports ln(1.6)."""
    projection = {(SEASON, p): 100.0 for p in range(5000, 5100)}
    for p in range(5000, 5050):
        projection[(SEASON, p)] = 160.0
    log: list[Transaction] = []
    for i in range(12):
        log.append(
            tx(
                "TRADE_PROPOSAL",
                1,
                3,
                [item("TRADE", 5000 + i, frm=2, to=1), item("TRADE", 5060 + i, frm=1, to=2)],
            )
        )
        log.append(
            tx(
                "TRADE_PROPOSAL",
                2,
                3,
                [item("TRADE", 5070 + i, frm=1, to=2), item("TRADE", 5080 + i, frm=2, to=1)],
            )
        )
    panel = panel_of(season_record(transactions=log))
    facts = facts_from(season_projection=projection)
    table, _ = endowment(panel, facts)
    greedy = table[panel.focus.owners[1]]
    fair = table[panel.focus.owners[2]]
    assert greedy.raw == pytest.approx(math.log(1.6), abs=1e-9)
    assert greedy.detail["implied_wta_wtp"] == pytest.approx(1.6)
    assert fair.raw == pytest.approx(0.0, abs=1e-9)


def test_trade_receptiveness_separates_a_wall_from_a_dealer() -> None:
    log: list[Transaction] = []
    for i in range(12):
        log.append(tx("TRADE_DECLINE", 1, 3, [], stamp=BASE_MS + i))
    for i in range(10):
        log.append(tx("TRADE_ACCEPT", 2, 3, [], stamp=BASE_MS + i))
    for team in range(3, 9):
        for i in range(5):
            kind = "TRADE_ACCEPT" if i < 3 else "TRADE_DECLINE"
            log.append(tx(kind, team, 3, [], stamp=BASE_MS + i))
    panel = panel_of(season_record(transactions=log))
    table, shrink = trade_receptiveness(panel)
    wall = table[panel.focus.owners[1]]
    dealer = table[panel.focus.owners[2]]
    assert wall.raw == 0.0 and dealer.raw == 1.0
    assert wall.estimate < shrink.league_mean < dealer.estimate
    assert wall.notable and dealer.notable

    report = behavioral_report(panel, facts_from(), permutation_reps=50)
    assert report.by_manager(panel.focus.owners[1]).counterparty == "unresponsive"
    assert report.by_manager(panel.focus.owners[2]).counterparty == "willing"


# --------------------------------------------------------------------------------------
# Draft-board biases
# --------------------------------------------------------------------------------------


def test_name_brand_finds_the_manager_who_drafts_fame() -> None:
    rng = np.random.default_rng(8)
    pool = list(range(6000, 6200))
    position = dict.fromkeys(pool, 3)
    projection = {(SEASON, p): float(rng.uniform(20, 300)) for p in pool}
    # ADP is merit plus an independent fame shock, so fame and merit really do diverge.
    fame = {p: projection[(SEASON, p)] + rng.normal(scale=90.0) for p in pool}
    order = sorted(pool, key=lambda p: -fame[p])
    adp = {(SEASON, p): float(i + 1) for i, p in enumerate(order)}

    picks: list[DraftPick] = []
    taken: set[int] = set()
    for overall in range(1, 8 * 14 + 1):
        team = (overall - 1) % 8 + 1
        free = [p for p in pool if p not in taken]
        target = (
            min(free, key=lambda p: adp[(SEASON, p)])
            if team == 1
            else max(free, key=lambda p: projection[(SEASON, p)])
        )
        taken.add(target)
        picks.append(pick(team, overall, target))

    panel = panel_of(season_record(picks=picks))
    facts = facts_from(season_projection=projection, adp=adp, position=position)
    table, _ = name_brand(panel, facts)
    star_chaser = table[panel.focus.owners[1]]
    assert star_chaser.estimable
    assert star_chaser.estimate > 0
    assert star_chaser.estimate > max(table[panel.focus.owners[t]].estimate for t in range(2, 9))


def test_home_team_uses_a_permutation_null_not_one_in_thirty_two() -> None:
    """A manager drafting at random must not be called stacked, and a stacker must be.

    The 1/32 baseline the brief suggests fails the first half of that: sixteen players
    drawn from an unevenly represented pool produce a modal share far above 1/32 by
    chance. Measured on the user's real 2026 drafts the permutation null sits at 0.143
    against 1/32 = 0.031, so 1/32 would have called every manager in every league 4.2x
    stacked. The test pins both directions.
    """
    rng = np.random.default_rng(9)
    pool = list(range(7000, 7200))
    # Twelve players on NFL team 7 so the stack is unmistakable, everybody else spread
    # over the remaining 31 teams at random -- unevenly, which is the whole point.
    pro_team = {(SEASON, p): int(rng.integers(8, 33)) for p in pool}
    stacker_players = pool[:12]
    for p in stacker_players:
        pro_team[(SEASON, p)] = 7
    picks: list[DraftPick] = []
    remaining = list(stacker_players)
    rest = [p for p in pool if p not in set(stacker_players)]
    cursor = 0
    for overall in range(1, 8 * 14 + 1):
        team = (overall - 1) % 8 + 1
        if team == 1 and remaining:
            target = remaining.pop()
        else:
            target = rest[cursor]
            cursor += 1
        picks.append(pick(team, overall, target))

    panel = panel_of(season_record(picks=picks))
    facts = facts_from(pro_team=pro_team)
    table, _ = home_team(panel, facts, reps=600)
    stacker = table[panel.focus.owners[1]]
    assert stacker.estimable
    assert stacker.detail["modal_team_id"] == 7.0
    assert stacker.detail["permutation_p"] < 0.05
    assert stacker.notable and stacker.vs_league > 0

    # Everyone else drafted at random and must come back unremarkable, even though their
    # modal share is comfortably above 1/32.
    others = [table[panel.focus.owners[t]] for t in range(2, 9)]
    assert not any(o.notable for o in others)
    assert all(o.detail["permutation_p"] > 0.05 for o in others)


# --------------------------------------------------------------------------------------
# Clock and activity
# --------------------------------------------------------------------------------------


def test_latency_ranks_the_fast_manager_first_and_ignores_waivers() -> None:
    """Teams act in a fixed order every week; the ordering must come back exactly.

    The waiver claims are all stamped at the batch time, which is what ESPN really does,
    and they must not move anybody: including them is how you conclude the whole league
    acts at 3am.
    """
    log: list[Transaction] = []
    batch = BASE_MS + 3 * 3_600_000
    for week in range(2, 12):
        base = BASE_MS + week * 7 * DAY_MS
        for offset, team in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
            log.append(
                tx(
                    "FREEAGENT",
                    team,
                    week,
                    [item("ADD", 8000 + week * 10 + team, to=team)],
                    stamp=base + offset * 3_600_000,
                )
            )
        for team in (8, 7, 6, 5, 4, 3, 2, 1):
            log.append(
                tx(
                    "WAIVER",
                    team,
                    week,
                    [item("ADD", 9000 + week * 10 + team, to=team)],
                    stamp=batch,
                )
            )
    panel = panel_of(season_record(transactions=log))
    table, shrink = latency(panel)
    values = [table[panel.focus.owners[t]].raw for t in range(1, 9)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert shrink.between_sd > 0
    # Free-agent counts only; the waiver claims contributed nothing.
    assert table[panel.focus.owners[1]].detail["free_agent_adds"] == 10.0


def test_quiet_hours_finds_the_window_nobody_uses() -> None:
    log = []
    for week in range(2, 12):
        base = BASE_MS + week * 7 * DAY_MS
        for hour in (14, 15, 16, 17, 18, 19, 20, 21):
            log.append(
                tx(
                    "FREEAGENT",
                    1,
                    week,
                    [item("ADD", 8500 + week * 100 + hour, to=1)],
                    stamp=base - (base % DAY_MS) + hour * 3_600_000,
                )
            )
    quiet = quiet_hours(panel_of(season_record(transactions=log)))
    assert all(count == 0 for _, count in quiet)


def test_activity_recovers_planted_rates_and_flags_a_dormant_manager() -> None:
    log: list[Transaction] = []
    per_week = {1: 4, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1, 7: 2, 8: 2}
    for week in range(1, FINAL_WEEK + 1):
        for team, moves in per_week.items():
            # Team 1 stops after week 8; everybody else keeps going.
            if team == 1 and week > 8:
                continue
            for i in range(moves):
                log.append(
                    tx(
                        "FREEAGENT",
                        team,
                        week,
                        [item("ADD", 9500 + week * 100 + team * 10 + i, to=team)],
                        stamp=BASE_MS + week * DAY_MS + i,
                    )
                )
    panel = panel_of(season_record(transactions=log))
    table, shrink = activity(panel)
    busy = table[panel.focus.owners[2]]
    quiet = table[panel.focus.owners[5]]
    # The raw rate recovers the plant; the published estimate is that pulled toward the
    # league, which is why the two are asserted separately.
    assert busy.detail["moves_per_week"] == pytest.approx(4.0, rel=0.05)
    assert quiet.detail["moves_per_week"] == pytest.approx(1.0, rel=0.1)
    assert math.exp(busy.estimate) > math.exp(quiet.estimate) * 2.5
    assert shrink.between_sd > 0

    gone = table[panel.focus.owners[1]]
    assert gone.detail["weeks_quiet"] == float(FINAL_WEEK - 8)
    assert gone.detail["dormancy_p"] < 0.05


def test_dormancy_is_not_asked_about_in_week_one() -> None:
    """Everyone is quiet the week after the draft; that is not evidence of anything."""
    log = [tx("FREEAGENT", 2, 1, [item("ADD", 9999, to=2)])]
    panel = panel_of(season_record(transactions=log, complete=False))
    table, _ = activity(panel)
    assert all(e.detail["dormancy_p"] == 1.0 for e in table.values() if e.detail)
    report = behavioral_report(panel, facts_from(), permutation_reps=50)
    assert all(p.counterparty != "dormant" for p in report.profiles)


# --------------------------------------------------------------------------------------
# Events, refusal and the report
# --------------------------------------------------------------------------------------


def test_roster_events_orders_the_draft_first_and_splits_a_trade() -> None:
    picks = [pick(1, 1, 100), pick(2, 2, 200)]
    log = [
        tx("FREEAGENT", 1, 2, [item("ADD", 300, to=1), item("DROP", 100, frm=1)]),
        tx("TRADE_ACCEPT", 2, 3, [item("TRADE", 200, frm=2, to=1)]),
    ]
    events = roster_events(season_record(picks=picks, transactions=log))
    assert [e.source for e in events][:2] == ["DRAFT", "DRAFT"]
    trade = [e for e in events if e.source == "TRADE"]
    assert {(e.direction, e.team_id) for e in trade} == {("drop", 2), ("add", 1)}


def test_roster_events_skip_commissioner_actions_and_failed_claims() -> None:
    log = [
        tx("WAIVER", 1, 2, [item("ADD", 400, to=1)], status="FAILED_INVALIDPLAYERSOURCE"),
        Transaction(
            id="lm",
            type="ROSTER",
            status="EXECUTED",
            execution_type="EXECUTE",
            scoring_period_id=2,
            team_id=1,
            member_id=None,
            bid_amount=0,
            is_pending=False,
            is_league_manager=True,
            proposed_date=BASE_MS,
            process_date=None,
            related_transaction_id=None,
            items=(item("DROP", 401, frm=1),),
        ),
    ]
    assert roster_events(season_record(transactions=log)) == ()


def test_an_unplayed_season_is_not_played_even_with_a_full_set_of_picks() -> None:
    """ESPN ships 192 placeholder picks for a league that was set up and abandoned."""
    ghost = SeasonRecord(
        league_id=1,
        season=2022,
        size=8,
        owners=owners(),
        names={},
        team_names={},
        picks=tuple(pick((i - 1) % 8 + 1, i, 100 + i) for i in range(1, 97)),
        transactions=(),
        final_week=FINAL_WEEK,
        complete=True,
        drafted=False,
    )
    assert not ghost.played
    assert season_record(picks=list(ghost.picks), transactions=[tx("ROSTER", 1, 2, [])]).played


def test_traits_are_refused_wholesale_when_there_is_only_a_draft() -> None:
    """The week-1 regime: a draft, no transactions, and six of nine traits must refuse."""
    rng = np.random.default_rng(21)
    pool = list(range(11000, 11200))
    position = dict.fromkeys(pool, 2)
    projection = {(SEASON, p): float(rng.uniform(10, 300)) for p in pool}
    prior = {(SEASON - 1, p): float(rng.uniform(10, 300)) for p in pool}
    adp = {(SEASON, p): float(i + 1) for i, p in enumerate(pool)}
    pro_team = {(SEASON, p): int(rng.integers(1, 33)) for p in pool}
    picks = [pick((i - 1) % 8 + 1, i, pool[i - 1]) for i in range(1, 8 * 14 + 1)]

    panel = panel_of(season_record(picks=picks, transactions=[], complete=False))
    facts = facts_from(
        season_actual=prior,
        season_projection=projection,
        adp=adp,
        position=position,
        pro_team=pro_team,
    )
    report = behavioral_report(panel, facts, permutation_reps=200)
    profile = report.profiles[0]
    assert set(profile.estimable) == {"draft_recency", "home_team", "name_brand"}
    for trait in ("add_recency", "sunk_cost", "endowment", "trade_receptiveness", "latency"):
        est = profile.traits[trait]
        assert not est.estimable
        assert math.isnan(est.estimate)
        assert est.required_n == MIN_N[trait]
    assert report.dead_traits


def test_report_shape_and_refusal_defaults() -> None:
    refusal = TraitEstimate(
        trait="sunk_cost",
        manager="m",
        n=1,
        required_n=20,
        estimable=False,
        raw=float("nan"),
        raw_stderr=float("nan"),
        estimate=float("nan"),
        stderr=float("nan"),
        weight=0.0,
        league_mean=float("nan"),
    )
    assert not refusal.distinct and not refusal.notable
    assert "needs 20" in refusal.describe()


def test_panel_requires_at_least_one_season() -> None:
    with pytest.raises(BehavioralError):
        BehavioralPanel(league_id=1, focus_season=SEASON, seasons=())


def test_load_player_facts_refuses_a_missing_corpus(tmp_path) -> None:
    with pytest.raises(BehavioralError):
        load_player_facts([SEASON], root=tmp_path / "nothing")


# --------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------

#: (league_id, season) for the user's real leagues. Only 161496047 has prior seasons --
#: the reverse of what the project brief assumed, and worth an assertion so the next
#: person to read this does not have to rediscover it.
LIVE_LEAGUES = ((272150391, 2026), (161496047, 2026), (634537479, 2026))


@pytest.mark.network
def test_live_panels_build_and_report() -> None:
    """End to end on the real leagues: the panel builds, the report is shaped, nothing lies."""
    from fantasy_quant.edges.behavioral import build_panel
    from fantasy_quant.pipeline import client_from_env

    client = client_from_env()
    try:
        for league_id, season in LIVE_LEAGUES:
            panel = build_panel(league_id, season, client=client)
            assert panel.focus.played
            assert len(panel.managers) == panel.focus.size
            # The 2022 archive answers 200 with 192 placeholder picks and must be absent.
            assert 2022 not in panel.history_seasons

            report = behavioral_report(panel, permutation_reps=300)
            assert len(report.profiles) == panel.focus.size
            buckets = (
                set(report.usable_traits)
                | set(report.marginal_traits)
                | set(report.confounded_traits)
                | set(report.dead_traits)
            )
            assert buckets == set(MIN_N)
            # The four buckets partition; a trait is in exactly one of them.
            assert len(buckets) == (
                len(report.usable_traits)
                + len(report.marginal_traits)
                + len(report.confounded_traits)
                + len(report.dead_traits)
            )
            for profile in report.profiles:
                for trait, est in profile.traits.items():
                    assert est.trait == trait
                    if est.estimable:
                        assert math.isfinite(est.estimate) and est.n >= est.required_n
                    else:
                        assert math.isnan(est.estimate) and math.isnan(est.raw)
                    # Nothing is ever "notable" on a trait the league cannot measure, or
                    # on one whose ranking is a re-description of something else.
                    assert not (est.notable and trait not in report.usable_traits)
                    assert not (est.notable and est.confounded)
                # An action is only ever emitted off a trait that separates this league,
                # off the autodraft share, or off a counterparty label -- and every
                # counterparty label needs a manager who is distinct from his own league.
                if profile.counterparty in ("willing", "unresponsive"):
                    assert profile.traits["trade_receptiveness"].distinct
                    assert report.separates("trade_receptiveness")
            # `rank` refuses any trait the league cannot order.
            for trait in report.dead_traits + report.confounded_traits:
                assert report.rank(trait) == ()
    finally:
        client.close()


@pytest.mark.network
def test_live_only_one_league_carries_history() -> None:
    """Pins the brief's error: Blacksburg has the archive, the other two do not."""
    from fantasy_quant.edges.behavioral import build_panel
    from fantasy_quant.pipeline import client_from_env

    client = client_from_env()
    try:
        with_history = build_panel(161496047, 2026, client=client)
        assert set(with_history.history_seasons) == {2021, 2023, 2024, 2025, 2026}
        for league_id in (272150391, 634537479):
            assert build_panel(league_id, 2026, client=client).history_seasons == (2026,)
    finally:
        client.close()


@pytest.mark.network
def test_live_waiver_timestamps_are_the_batch_run_not_the_manager() -> None:
    """The measurement that forces `latency` to read free agents only.

    If this ever stops holding -- if ESPN starts serving the submission time on an
    executed claim -- `latency` gains four times its current sample and should be
    rewritten to use it.
    """
    from datetime import UTC, datetime

    from fantasy_quant.espn.league import League
    from fantasy_quant.pipeline import client_from_env

    client = client_from_env()
    try:
        waiver_hours: list[int] = []
        free_agent_hours: list[int] = []
        for season in (2021, 2023, 2024, 2025):
            for t in League(client, 161496047, season).transactions():
                if t.status != "EXECUTED" or not t.proposed_date:
                    continue
                hour = datetime.fromtimestamp(t.proposed_date / 1000, UTC).hour
                if t.type == "WAIVER":
                    waiver_hours.append(hour)
                elif t.type == "FREEAGENT":
                    free_agent_hours.append(hour)
    finally:
        client.close()

    assert len(waiver_hours) > 300 and len(free_agent_hours) > 300
    batch = sum(1 for h in waiver_hours if 7 <= h <= 9) / len(waiver_hours)
    assert batch > 0.9, f"executed waivers no longer cluster in the batch window: {batch:.2f}"
    assert len(set(waiver_hours)) <= 6
    assert len(set(free_agent_hours)) >= 20


# --------------------------------------------------------------------------------------
# Regressions: four ways this module published noise on the user's real leagues
#
# Every test below reproduces something the module actually emitted on 2026-09-07, and
# every one of them passed the original suite. They are the difference between "the
# estimator recovers a planted effect" and "the estimator refuses an unplanted one".
# --------------------------------------------------------------------------------------


def _censored_holds_league(
    *, patience: dict[int, float], sunk_effect: float = 0.5, seasons: int = 3, seed: int = 31
) -> tuple[BehavioralPanel, PlayerFacts]:
    """Every manager has the SAME sunk-cost effect; they differ only in how fast they cut.

    Hold duration is censored at the end of the season, so a manager patient enough to keep
    everyone has a constant outcome and an OLS round slope of zero *by arithmetic*. This
    generator makes that the only difference between managers, which is what the real
    Blacksburg log looks like: the fitted round coefficient correlates +0.87 with how often
    a manager drops anybody at all.
    """
    rng = np.random.default_rng(seed)
    records: list[SeasonRecord] = []
    season_actual: dict[tuple[int, int], float] = {}
    week_actual: dict[tuple[int, int, int], float] = {}
    position: dict[int, int] = {}

    player = 20_000
    for s in range(seasons):
        season = SEASON - s
        picks: list[DraftPick] = []
        drops: list[Transaction] = []
        for overall in range(1, 8 * 12 + 1):
            team = (overall - 1) % 8 + 1
            round_id = (overall - 1) // 8 + 1
            player += 1
            position[player] = 2
            quality = 12.0 - 0.7 * round_id + rng.normal(scale=2.0)
            season_actual[(season, player)] = max(quality, 0.0) * FINAL_WEEK
            for w in range(1, 5):
                week_actual[(season, w, player)] = max(quality, 0.0)
            picks.append(pick(team, overall, player, round_id=round_id))
            # One shared law of motion, shifted by the manager's own patience.
            held = patience[team] + 0.9 * quality - sunk_effect * round_id + rng.normal(scale=1.0)
            held = int(np.clip(round(held), 1, FINAL_WEEK + 1))
            if held <= FINAL_WEEK:
                drops.append(tx("ROSTER", team, held + 1, [item("DROP", player, frm=team)]))
        records.append(season_record(picks=picks, transactions=drops, season=season))
    return panel_of(*records), facts_from(
        season_actual=season_actual, week_actual=week_actual, position=position
    )


def test_sunk_cost_refuses_to_rank_managers_who_differ_only_in_how_fast_they_cut() -> None:
    """The censoring confound, which the production control does not touch.

    On the real league this shipped as advice with the sign reversed: the manager who
    released the *fewest* of his own draft picks -- 21%, the lowest rate in the league, and
    the lowest activity too -- was published as "cuts his own draft picks: his early-round
    busts reach the wire", because his flat outcome gave him a flat round slope. Here every
    manager has an identical sunk-cost effect and they differ only in patience, so any
    per-manager ranking the estimator produces is the confound and nothing else.
    """
    patience = {team: 6.0 + 1.5 * (team - 1) for team in range(1, 9)}
    panel, facts = _censored_holds_league(patience=patience)
    table, shrink, fits = sunk_cost(panel, facts)

    released = {m: f.released_share for m, f in fits.items()}
    coefficients = {m: f.controlled for m, f in fits.items()}
    keys = list(released)
    corr = float(np.corrcoef([released[m] for m in keys], [coefficients[m] for m in keys])[0, 1])
    assert corr > 0.6, f"the generator must produce the confound to be a test of it: {corr}"

    # It must be measured, published, and acted on.
    assert shrink.confound, "a measured censoring confound has to reach the caller"
    assert all(e.confounded for e in table.values() if e.estimable)
    assert not any(e.notable for e in table.values())
    assert not shrink.separates
    for est in table.values():
        if est.estimable:
            assert est.detail["censoring_corr"] == pytest.approx(corr, abs=1e-9)
            assert 0.0 <= est.detail["released_share"] <= 1.0

    report = behavioral_report(panel, facts, permutation_reps=50)
    assert "sunk_cost" in report.confounded_traits
    assert "sunk_cost" not in report.usable_traits
    assert report.rank("sunk_cost") == ()
    # And most concretely: the most patient manager in the league -- the one who lets go of
    # nobody -- must never be told he lets go of people sooner than his league does.
    patient = panel.focus.owners[8]
    profile = report.by_manager(patient)
    assert profile is not None
    assert not any("draft picks" in a for a in profile.actions())


def test_notable_requires_the_trait_to_separate_anyone_at_all() -> None:
    """A manager whose own standard error is small does not get to rank a blind trait.

    This is Type shi's `draft_recency` exactly: eleven managers measured at se ~0.05 and one
    at 0.008 because his thirteen picks happened to miss the distribution's tail. The report
    classified the trait `marginal` -- "read, do not act" -- and `actions()` then acted on
    it, emitting "drafts last season's leaderboard: sell him last year's name" off a raw
    tilt of +0.036.
    """
    ordinary = [-0.039, -0.011, 0.100, 0.040, -0.099, -0.000, -0.037, 0.009, -0.097, 0.015, 0.014]
    raw = {f"m{i}": v for i, v in enumerate(ordinary)}
    stderr = {f"m{i}": 0.05 for i in range(len(ordinary))}
    raw["lucky"], stderr["lucky"] = 0.09, 0.008
    counts = dict.fromkeys(raw, 14)

    table, shrink = empirical_bayes("draft_recency", raw, stderr, counts, required_n=1)
    assert 0 < shrink.signal_to_noise < 1.0, shrink.signal_to_noise
    lucky = table["lucky"]
    # He really is separated in the measurement sense -- that was never the problem.
    assert lucky.distinct
    assert not lucky.separating
    assert not lucky.notable, "a trait blinder than its own error cannot rank anybody"
    assert not any(e.notable for e in table.values())


def test_a_lucky_manager_does_not_get_a_standard_error_of_zero() -> None:
    """The tilt distribution is a spike plus a tail, and whoever misses the tail looks exact.

    On Type shi's real 2026 draft one manager's thirteen picks happened to include none of
    the ~12% of players coming off a lost season, so his own `sd/sqrt(n)` came out at 0.0083
    against 0.15-0.30 for everyone else. Empirical Bayes handed him almost all of his own
    weight and the module published "drafts last season's leaderboard: sell him last year's
    name" off a raw tilt of +0.036. Here every manager follows the identical rule -- take the
    best projection left -- so any difference between them is which players fell to them.
    """
    rng = np.random.default_rng(41)
    pool = list(range(30_000, 30_200))
    position = dict.fromkeys(pool, 2)
    projection = {(SEASON, p): float(rng.uniform(20, 300)) for p in pool}
    prior = {(SEASON - 1, p): projection[(SEASON, p)] for p in pool}
    for p in rng.choice(pool, size=24, replace=False):
        prior[(SEASON - 1, int(p))] = float(rng.uniform(0, 15))

    picks: list[DraftPick] = []
    taken: set[int] = set()
    for overall in range(1, 8 * 14 + 1):
        team = (overall - 1) % 8 + 1
        free = [p for p in pool if p not in taken]
        target = max(free, key=lambda p: projection[(SEASON, p)])
        taken.add(target)
        picks.append(pick(team, overall, target))

    panel = panel_of(season_record(picks=picks, transactions=[], complete=False))
    facts = facts_from(season_actual=prior, season_projection=projection, position=position)
    table, _ = draft_recency(panel, facts)

    estimable = [e for e in table.values() if e.estimable]
    assert len(estimable) == 8
    lucky = min(estimable, key=lambda e: e.detail["own_sd"])
    assert lucky.detail["own_sd"] < 1e-6, "the generator must produce a zero-variance manager"
    # Unpooled his standard error was the 1e-9 floor, which carries essentially all of his
    # own weight through shrinkage and makes him the most confident number in the league.
    assert lucky.raw_stderr > 0.3 * lucky.detail["pooled_sd"] / math.sqrt(lucky.n)
    assert lucky.raw_stderr > 0.02

    report = behavioral_report(panel, facts, permutation_reps=100)
    assert "draft_recency" not in report.usable_traits
    for profile in report.profiles:
        assert "draft_recency" not in profile.notable
        assert not any("last year" in a or "leaderboard" in a for a in profile.actions())


def test_counterparty_refuses_a_label_the_interval_does_not_support() -> None:
    """Three managers at 0.565, 0.573 and 0.576 were "willing"; one at 0.463 was not.

    Every one of those intervals straddled the 0.5 cutoff, and dropping any single season
    from the real panel moved the "willing" count between one and five. A threshold on a
    shrunk estimate is not a classification.
    """
    log: list[Transaction] = []
    # Six managers clustered around the middle, and one genuine wall to give the trait a
    # real between-manager spread so this is not passing by collapse.
    middling = {3: (6, 4), 4: (5, 4), 5: (6, 5), 6: (4, 4), 7: (5, 5), 8: (6, 6)}
    for team, (accepts, declines) in middling.items():
        for i in range(accepts):
            log.append(tx("TRADE_ACCEPT", team, 3, [], stamp=BASE_MS + i))
        for i in range(declines):
            log.append(tx("TRADE_DECLINE", team, 3, [], stamp=BASE_MS + 100 + i))
    for i in range(16):
        log.append(tx("TRADE_DECLINE", 1, 3, [], stamp=BASE_MS + i))
    for i in range(16):
        log.append(tx("TRADE_ACCEPT", 2, 3, [], stamp=BASE_MS + i))

    panel = panel_of(season_record(transactions=log))
    report = behavioral_report(panel, facts_from(), permutation_reps=50)
    table = {m: report.by_manager(m).traits["trade_receptiveness"] for m in panel.managers}

    assert not report.shrinkage["trade_receptiveness"].collapsed
    for team in middling:
        manager = panel.focus.owners[team]
        estimate = table[manager]
        low, high = estimate.ci95
        assert low < 0.5 < high, "the generator must put these on the knife edge"
        assert report.by_manager(manager).counterparty == "unknown"
    # The two managers who really are separated keep their labels.
    assert report.by_manager(panel.focus.owners[1]).counterparty == "unresponsive"
    assert report.by_manager(panel.focus.owners[2]).counterparty == "willing"


def test_rank_refuses_a_trait_that_cannot_tell_managers_apart() -> None:
    """A collapsed trait gives everyone the same number; sorting that is franchise order."""
    log = [
        tx("FREEAGENT", team, week, [item("ADD", 40_000 + week * 10 + team, to=team)])
        for week in range(1, FINAL_WEEK + 1)
        for team in range(1, 9)
    ]
    panel = panel_of(season_record(transactions=log))
    report = behavioral_report(panel, facts_from(), permutation_reps=50)
    assert report.shrinkage["activity"].collapsed
    assert not report.separates("activity")
    assert report.rank("activity") == ()
    assert report.rank("name_brand") == ()


def test_activity_standard_errors_admit_that_moves_arrive_in_bursts() -> None:
    """Roster moves are overdispersed: the Poisson error is too small and says so.

    Measured on the real league the Pearson dispersion across manager-seasons is 2.87, so
    the honest standard error is about 1.7x the Poisson one and the published
    signal-to-noise falls from 6.19 to 3.55. Two managers here make the same number of
    moves; one spreads them evenly across seasons and one does them all in one.
    """
    records: list[SeasonRecord] = []
    for s, season in enumerate((SEASON - 2, SEASON - 1, SEASON)):
        log: list[Transaction] = []
        for week in range(1, FINAL_WEEK + 1):
            for i in range(2):  # steady manager: two a week, every season
                log.append(
                    tx(
                        "FREEAGENT",
                        1,
                        week,
                        [item("ADD", 50_000 + season * 1000 + week * 10 + i, to=1)],
                        stamp=BASE_MS + week * DAY_MS + i,
                    )
                )
            if s == 1:  # bursty manager: the same total, all in one season
                for i in range(6):
                    log.append(
                        tx(
                            "FREEAGENT",
                            2,
                            week,
                            [item("ADD", 60_000 + season * 1000 + week * 10 + i, to=2)],
                            stamp=BASE_MS + week * DAY_MS + i,
                        )
                    )
        records.append(season_record(transactions=log, season=season))
    panel = panel_of(*records)
    table, _ = activity(panel)
    steady = table[panel.focus.owners[1]]
    bursty = table[panel.focus.owners[2]]

    assert steady.detail["moves"] == bursty.detail["moves"]
    assert steady.detail["moves_per_week"] == pytest.approx(bursty.detail["moves_per_week"])
    assert steady.detail["dispersion"] == pytest.approx(1.0, abs=0.3)
    assert bursty.detail["dispersion"] > 3.0
    # Same rate, same count, and the bursty manager is measured worse. Under the old
    # Poisson 1/sqrt(count) the two were identical to the last decimal.
    assert bursty.raw_stderr > 1.5 * steady.raw_stderr


def test_report_names_a_season_the_corpus_never_captured() -> None:
    """ESPN serves the season in full, the corpus has never seen it, and the traits skip it.

    On the real league this is 2021: 860 transactions and 192 picks that fed nothing to five
    of the nine traits, while the report said "history [2021, 2023, 2024, 2025, 2026]".
    """
    old = season_record(
        picks=[pick((i - 1) % 8 + 1, i, 70_000 + i) for i in range(1, 97)],
        transactions=[tx("FREEAGENT", 1, 2, [item("ADD", 70_500, to=1)])],
        season=SEASON - 1,
    )
    new = season_record(
        picks=[pick((i - 1) % 8 + 1, i, 80_000 + i) for i in range(1, 97)],
        transactions=[tx("FREEAGENT", 1, 2, [item("ADD", 80_500, to=1)])],
        season=SEASON,
    )
    facts = facts_from(
        season_actual={(SEASON, 80_000 + i): 100.0 for i in range(1, 97)},
        position={80_000 + i: 2 for i in range(1, 97)},
    )
    report = behavioral_report(panel_of(old, new), facts, permutation_reps=50)
    assert report.seasons_used == (SEASON - 1, SEASON)
    assert report.seasons_without_football == (SEASON - 1,)
    assert "NO CORPUS RESULTS" in report.summary()
