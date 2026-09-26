"""Fetch real games from the public Chess.com API to build a distillation
corpus, instead of hand-transcribing famous games (an easy way to introduce
illegal-move typos that silently break python-chess parsing downstream).

No auth needed for public games. Chess.com's Published-Data API organizes a
player's games into monthly archives; this pulls the most recent archives
first and stops once --max-games is reached.

Usage:
    python -m distill.fetch_chesscom_games --username shanky8991 \\
        --max-games 60 --out-dir examples/distill_games

    # multiple sources for more diversity:
    python -m distill.fetch_chesscom_games --username shanky8991,hikaru \\
        --max-games 40 --out-dir examples/distill_games

Docs: https://www.chess.com/news/view/published-data-api
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ARCHIVES_URL = "https://api.chess.com/pub/player/{username}/games/archives"

# Chess.com's API docs ask integrators to identify themselves via User-Agent
# rather than blocking on API keys. Harmless to leave as-is; feel free to
# add your own contact info.
HEADERS = {"User-Agent": "chess-annotator-distillation-script (personal project)"}


def fetch_archive_urls(username: str, timeout: int = 30) -> list:
    import requests

    resp = requests.get(ARCHIVES_URL.format(username=username), headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    # Oldest-to-newest; we want newest first so we grab recent games.
    return list(reversed(resp.json().get("archives", [])))


def fetch_games_from_archive(archive_url: str, rules: str = "chess", timeout: int = 30, time_classes=None) -> list:
    import requests

    resp = requests.get(archive_url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    games = resp.json().get("games", [])
    pgns = []
    for g in games:
        pgn = g.get("pgn")
        if not pgn:
            continue
        # Skip variants (chess960, bughouse, etc.) -- the rest of the
        # pipeline (python-chess analysis) assumes standard chess rules.
        if rules and g.get("rules", "chess") != rules:
            continue
        if time_classes and g.get("time_class") not in time_classes:
            continue
        pgns.append(pgn)
    return pgns


def fetch_user_games(username: str, max_games: int = 50, rules: str = "chess",
                      sleep_between_archives: float = 0.5, time_classes=None) -> str:
    archive_urls = fetch_archive_urls(username)
    collected = []
    for url in archive_urls:
        if len(collected) >= max_games:
            break
        pgns = fetch_games_from_archive(url, rules=rules, time_classes=time_classes)
        collected.extend(pgns)
        time.sleep(sleep_between_archives)
    collected = collected[:max_games]
    return "\n\n".join(collected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True, help="Comma-separated Chess.com username(s)")
    ap.add_argument("--max-games", type=int, default=50, help="Max games per username")
    ap.add_argument("--out-dir", default="examples/distill_games")
    ap.add_argument("--rules", default="chess", help="Game variant to keep (default: standard chess only)")
    ap.add_argument("--time-classes", default="", help="Comma-separated Chess.com time classes to keep, e.g. 'rapid,blitz' (default: all)")
    ap.add_argument("--sleep-between-users", type=float, default=2.0, help="Seconds between per-user requests (be polite)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usernames = [u.strip() for u in args.username.split(",") if u.strip()]
    for i, username in enumerate(usernames):
        print(f"Fetching up to {args.max_games} games for '{username}'...")
        tcs = {t.strip() for t in args.time_classes.split(",") if t.strip()} or None
        try:
            pgn_text = fetch_user_games(username, max_games=args.max_games, rules=args.rules, time_classes=tcs)
        except Exception as e:  # noqa: BLE001 -- a wrong username shouldn't stop the others
            print(f"  skipped '{username}': {e}")
            continue
        n_games = pgn_text.count("[Event ")
        out_path = out_dir / f"{username}.pgn"
        out_path.write_text(pgn_text, encoding="utf-8")
        print(f"  wrote {n_games} games -> {out_path}")
        if i < len(usernames) - 1:
            time.sleep(args.sleep_between_users)

    print(f"\nDone. PGN files are in {out_dir}/ -- generate_dataset.py reads every .pgn file there.")


if __name__ == "__main__":
    main()
