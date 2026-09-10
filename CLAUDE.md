# Working in this repo

## What this is

Multi-league decision analytics for real-money ESPN fantasy football. Every surface reports
**Δ P(championship)** so a waiver claim in one league is comparable to a lineup change in
another. Python 3.13 / uv / polars / numpy / scipy, FastAPI + Vite/React dashboard.

Read `docs/AUDIT.md` first — it is the live status of the correctness work, with measurements
and the ranked list of what is still open. `docs/PLAN.md` is the original design, `docs/RESEARCH.md`
the verified endpoints and measured constants.

## Hard constraints

- **Read-only against ESPN. Recommend only — never execute a transaction.**
- **The repo is public.** `config/leagues.toml` (real league names), `/data/` (the corpus) and
  `data/manual/` (ETR exports) are git-ignored and must stay that way.
- **Credentials.** `ESPN_SWID` and `ESPN_S2` live in a git-ignored `.env` at mode 600. Never
  print, log, or commit their values.
- **Establish the Run CSVs are paid subscriber content** and their ToS prohibits automated
  scraping. Manual CSV download into `data/manual/etr/` only — do not build a scraper.
- **The API binds to 127.0.0.1** and must refuse non-loopback peers whatever the bind address;
  it holds session cookies.
- Test fixtures anonymise leaguemates' team names before being committed.

## How to work here

The recurring bug shape in this codebase is **a component computing something
defensible-looking that is not the right quantity**. Tests do not catch it, because the code
does correctly what it was written to do. Two things that do catch it:

- **Suspect broken invariants over wrong numbers.** A league-specific replacement floor coming
  back as an identical 4.704 in all three leagues was the tell that found the biggest bug —
  visible long before anyone measured the floor itself.
- **Ask what question a component should answer before asking whether its code is right.**

So:

1. **Measure on the three real leagues, not just the fixture, and not by argument.**
   `Registry.load().active()` gives all three. Field names are `cfg.team_id` and
   `cfg.scoring_variant` (not `my_team_id` / `variant`).
2. **Every change ships a negative control** — run the new path with the new input disabled and
   assert the output is byte-identical to before. This is what proves a change did what it
   claimed *and nothing else*.
3. **Pick the acceptance criterion from what the system consumes.** The season simulator
   consumes the conditional *mean*, so held-out **bias** is the criterion and MAE is the
   report. At K and D/ST the two give opposite answers; see `projections/calibration.py`.
4. **Report scope honestly, including when a fix turns out to be latent.** The title surrogate
   has no caller in `src/`; saying so was more useful than claiming impact.
5. **Prefer making an existing adaptive mechanism correct over adding a switch.** Arbitrary
   rules layered on top ("declare which positions you stream") are the wrong shape here.
6. Findings that are genuine design tradeoffs get **written down as tradeoffs**, not silently
   "fixed". Anything that moves a gate the user relies on (`trades.PLAYOFF_WEIGHT`, say) is a
   decision to bring back to them, not a patch to slip in.

## Commands

```
uv run pytest -q -m "not network"     # ~1,880 offline tests, ~70s
uv run pytest -q -m network           # hits ESPN and FanDuel with real credentials, ~4.5 min
uv run ruff check src tests           # before every commit
npm --prefix web run build            # after touching the dashboard
cd web && npx tsc --noEmit
uv run fq odds | waivers | queue | stream | trades | lineup | weekly
```

Live-market tests (FanDuel) are inherently a little flaky — judge them on distributions, not
on single items. Three of them were rewritten for exactly this reason; see `7ea0553`.

## Commit style

Commits explain **what was wrong and how it was measured**, not what files changed. Lead with
the defect, quote the offending line, give the numbers, and name the negative control. Look at
`git log` for the pattern.
