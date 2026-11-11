#!/usr/bin/env python3
"""Roblox Up-and-Coming forecast entry (make.fun roblox-up-and-coming-forecast-01).

Model: window-averaged empirical place distribution.
- Fit: for each historical read at time t and each game present, the fraction of
  reads in (t, t + H] (H = scored-window length from task.json) in which the game
  held each place 1..10, keyed by (rank bucket, velocity bucket). Velocity is the
  game's rank change over the previous ~12h (up / flat / down).
- Apply: look the table up for every game in the last chart, mix 0.65 with a
  persistence prior (mass near its current rank), cap each place's named sum at
  0.75 (the rest is left unnamed and spreads over the read's unnamed games),
  and order top10 by P(in top 10), then by expected place.
- Age-out: the chart only lists games published within the last 28 days. A game
  whose `created` date (from details, when the history carries it) is more than
  24 days before the window start has its mass decayed linearly to zero at
  28 days; when only first-seen is known, the same rule uses first-seen age.
- The chart's sort is computed, not a live count, and raw playing growth does
  not predict rank moves (measured correlation ~0.01 on the practice history),
  so playing counts are used only through ranks.

Deterministic: sorted iteration everywhere, no randomness, no wall clock.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output). Python 3.13, standard library only.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIX_TABLE = 0.65          # weight of the fitted table vs the persistence prior
PLACE_CAP = 0.75          # max named probability mass per place
MAX_ROWS_PER_PLACE = 100
MIN_P = 0.002             # drop rows below this
CANDIDATE_TOP = 60        # ranks of the last chart to consider
DROPOUT_TOP = 30          # ranks of recently-dropped games to consider
VEL_LAG_H = 6             # velocity measured against a read at least this old
AGE_DECAY_START = 24.0    # days; decay starts here
AGE_MAX = 28.0            # days; chart only lists games published within this


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def rank_bucket(rank):
    if rank == 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 15:
        return "11-15"
    if rank <= 20:
        return "16-20"
    if rank <= 30:
        return "21-30"
    if rank <= 50:
        return "31-50"
    return "51+"


def vel_bucket(delta):
    # delta = rank improvement (positive = climbed) over the velocity lag
    if delta >= 3:
        return "up"
    if delta <= -3:
        return "down"
    return "flat"


def prior_dist(rank):
    """Persistence prior: mass concentrated near the current rank."""
    dist = {}
    if rank <= 10:
        for k in range(1, 11):
            dist[k] = 0.55 * (0.45 ** abs(k - rank))
    else:
        for k in range(1, 11):
            dist[k] = 0.02 * (0.7 ** (rank - 10)) * (1.0 if k >= 6 else 0.4)
    return dist


def load_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            chart = row.get("up_and_coming")
            if not chart or not isinstance(chart, list):
                continue
            try:
                t = parse_time(row["at"])
            except (KeyError, ValueError, TypeError):
                continue
            rank_map = {}
            for g in chart:
                try:
                    rank_map[int(g["universe_id"])] = int(g["rank"])
                except (KeyError, ValueError, TypeError):
                    continue
            if rank_map:
                rows.append((t, rank_map))
    rows.sort(key=lambda item: item[0])
    return rows


def load_created(path, rows):
    """Map universe_id -> created timestamp, from details when present.
    Details may be a dict {uid: {...}} or a list of objects with universe_id.
    Falls back to first-seen time in the history."""
    created = {}
    first_seen = {}
    for t, rank_map in rows:
        for uid in rank_map:
            if uid not in first_seen:
                first_seen[uid] = t
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            details = row.get("details")
            if not details:
                continue
            items = []
            if isinstance(details, dict):
                for key, val in details.items():
                    if isinstance(val, dict):
                        items.append((key, val))
            elif isinstance(details, list):
                for val in details:
                    if isinstance(val, dict) and "universe_id" in val:
                        items.append((val["universe_id"], val))
            for key, val in items:
                try:
                    uid = int(key)
                except (ValueError, TypeError):
                    continue
                made = val.get("created")
                if not made:
                    continue
                try:
                    created[uid] = parse_time(made)
                except (ValueError, TypeError):
                    continue
    return created, first_seen


def fit_table(rows, horizon_s):
    """table[(rank_bucket, vel_bucket)][place] = mean fraction of reads in the
    next horizon_s where a game with those features held that place."""
    agg = defaultdict(lambda: defaultdict(float))
    count = defaultdict(int)
    n = len(rows)
    for i in range(n):
        t, rank_map = rows[i]
        # future reads inside the horizon
        fut = []
        for j in range(i + 1, n):
            if rows[j][0] <= t + horizon_s:
                fut.append(rows[j][1])
            else:
                break
        if len(fut) < 3:
            continue
        # per-game rank counts over the future reads (one pass)
        future_counts = defaultdict(Counter)
        for fmap in fut:
            for uid, rk in fmap.items():
                if rk <= 10:
                    future_counts[uid][rk] += 1
        # previous read at least VEL_LAG_H old, for velocity
        prev_map = None
        for j in range(i - 1, -1, -1):
            if t - rows[j][0] >= VEL_LAG_H * 3600:
                prev_map = rows[j][1]
                break
        nfut = float(len(fut))
        for uid, rk in rank_map.items():
            delta = 0
            if prev_map is not None and uid in prev_map:
                delta = prev_map[uid] - rk
            key = (rank_bucket(rk), vel_bucket(delta))
            count[key] += 1
            fc = future_counts.get(uid)
            for k in range(1, 11):
                agg[key][k] += (fc.get(k, 0) / nfut) if fc else 0.0
    table = {}
    for key, places in agg.items():
        m = count[key]
        table[key] = {k: v / m for k, v in places.items()}
    return table


def build_forecast(rows, created, first_seen, task):
    places_n = int(task.get("places", 10))
    window = task.get("window", {})
    horizon_s = 72.0 * 3600.0
    try:
        t_from = parse_time(window["from"])
        t_until = parse_time(window["until"])
        horizon_s = max(t_until - t_from, 3600.0)
    except (KeyError, ValueError, TypeError):
        t_from = rows[-1][0] if rows else 0.0
    table = fit_table(rows, horizon_s)

    t_last, last_map = rows[-1]
    # previous read at least VEL_LAG_H old, for velocity
    prev_map = None
    for t, rmap in reversed(rows[:-1]):
        if t_last - t >= VEL_LAG_H * 3600:
            prev_map = rmap
            break

    # candidate games: last chart's top ranks + recently dropped-out games
    cand = {}
    for uid, rk in last_map.items():
        if rk <= CANDIDATE_TOP:
            cand[uid] = rk
    for _, rmap in rows[-3:-1]:
        for uid, rk in rmap.items():
            if rk <= DROPOUT_TOP and uid not in cand:
                cand[uid] = 99

    # age-out decay factor per game
    decay = {}
    for uid in cand:
        born = created.get(uid)
        if born is None:
            born = first_seen.get(uid, t_from)
        age_days = (t_from - born) / 86400.0
        if age_days <= AGE_DECAY_START:
            decay[uid] = 1.0
        elif age_days >= AGE_MAX:
            decay[uid] = 0.0
        else:
            decay[uid] = (AGE_MAX - age_days) / (AGE_MAX - AGE_DECAY_START)

    probs = defaultdict(dict)  # place -> uid -> p
    for uid, rk in cand.items():
        delta = 0
        if prev_map is not None and uid in prev_map:
            delta = prev_map[uid] - rk
        tab = table.get((rank_bucket(rk), vel_bucket(delta)))
        pri = prior_dist(rk)
        d = decay[uid]
        for k in range(1, places_n + 1):
            tv = tab.get(k, 0.0) if tab else 0.0
            pv = pri.get(k, 0.0)
            probs[k][uid] = (MIX_TABLE * tv + (1.0 - MIX_TABLE) * pv) * d

    out_rows = []
    for k in range(1, places_n + 1):
        items = sorted(probs[k].items(), key=lambda kv: (-kv[1], kv[0]))
        items = [(u, p) for u, p in items if p > MIN_P][:MAX_ROWS_PER_PLACE]
        total = sum(p for _, p in items)
        if total > PLACE_CAP:
            items = [(u, p * PLACE_CAP / total) for u, p in items]
        for uid, p in items:
            out_rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    out_rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: highest P(in top 10), then lowest expected place, then lowest uid
    scored = []
    for uid in cand:
        tot = 0.0
        exp = 0.0
        for k in range(1, places_n + 1):
            p = probs[k].get(uid, 0.0)
            tot += p
            exp += k * p
        scored.append((uid, tot, exp / tot if tot > 0 else 99.0))
    scored.sort(key=lambda item: (-item[1], item[2], item[0]))
    top10 = [uid for uid, _, _ in scored[:places_n]]
    return {"probabilities": out_rows, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    rows = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not rows:
        raise SystemExit("no usable history reads")
    created, first_seen = load_created(os.path.join(INPUT_DIR, "history.ndjson"), rows)
    forecast = build_forecast(rows, created, first_seen, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
