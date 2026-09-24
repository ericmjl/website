# /// script
# requires-python = ">=3.11"
# dependencies = ["python-substack>=0.8.0"]
# ///
"""Substack draft + schedule CLI for the buffer-substack-sync skill.

Wraps the python-substack package (unofficial Substack API) so the
buffer-substack-sync skill can inventory, create, and schedule Substack
posts without a browser.

Commands:
  whoami                     Auth smoke test. Prints publication + user id.
  drafts [--limit N]         List drafts: id, title, schedule state, blog slugs in body.
  published [--limit N]      List published posts: id, title, url, post_date.
  create --title T --body-file F [--subtitle S] [--slug SLUG]
         [--schedule-at ISO] [--yes]
                             Create a draft from a markdown file. Without --yes,
                             prints the plan only. With --schedule-at, schedules it.
  schedule --draft-id ID --at ISO [--yes]
                             Schedule an existing draft.
  unschedule --draft-id ID [--yes]
                             Remove a draft's schedule.

Auth (in order):
  1. env: SUBSTACK_EMAIL + SUBSTACK_PASSWORD
     (optional SUBSTACK_PUBLICATION_URL, default https://dspn.substack.com)
  2. 1Password item "Substack" (fields: username, password) via the `op` CLI.

Secrets are never echoed; only their lengths are printed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BLOG_URL_RE = re.compile(
    r"ericmjl\.github\.io/blog/(\d{4})/(\d{1,2})/(\d{1,2})/([a-z0-9-]+)"
)
DEFAULT_PUBLICATION = "https://dspn.substack.com"
DEFAULT_COOKIES_PATH = "~/.config/buffer-substack-sync/cookies.json"


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_cookies(path: str) -> str:
    """Return a temp-file path holding a flat {name: value} cookie dict.

    requests' cookiejar only accepts flat dicts; browser exports are lists of
    dicts, so normalize either shape.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return path
    if isinstance(data, list):
        flat = {c["name"]: c["value"] for c in data if c.get("name") and c.get("value")}
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(flat, tmp)
        tmp.close()
        os.chmod(tmp.name, 0o600)
        return tmp.name
    die(f"cookies file {path} is neither a dict nor a list of dicts")


def creds() -> tuple[str, str, str]:
    pub = pub_url()
    email = os.environ.get("SUBSTACK_EMAIL")
    password = os.environ.get("SUBSTACK_PASSWORD")
    if email and password:
        print(
            f"auth: env credentials (username {len(email)} chars, "
            f"password {len(password)} chars) -> {pub}",
            file=sys.stderr,
        )
        return email, password, pub

    def op(field: str) -> str:
        result = subprocess.run(
            ["op", "item", "get", "Substack", "--fields", field],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            die(
                f"1Password read failed for field '{field}': {result.stderr.strip()}. "
                "Unlock 1Password (run any `op` command once) and retry."
            )
        return result.stdout.strip()

    email = op("username")
    password = op("password")
    print(
        f"auth: 1Password item 'Substack' (username {len(email)} chars, "
        f"password {len(password)} chars) -> {pub}",
        file=sys.stderr,
    )
    return email, password, pub


def client():
    """Build an authenticated Api client.

    Auth order: cookies file (env SUBSTACK_COOKIES_PATH or
    ~/.config/buffer-substack-sync/cookies.json), then email+password login
    (env, else 1Password). NOTE: Substack gates password logins behind a
    captcha, so the cookie file is the reliable path; the password login is
    a fallback that usually fails with "Please complete the captcha".
    """
    from substack import Api

    cookie_path = os.environ.get("SUBSTACK_COOKIES_PATH", DEFAULT_COOKIES_PATH)
    cookie_path = os.path.expanduser(cookie_path)
    if os.path.exists(cookie_path):
        print(f"auth: cookies file {cookie_path}", file=sys.stderr)
        normalized = normalize_cookies(cookie_path)
        return Api(cookies_path=normalized, publication_url=pub_url(), timeout=60)

    email, password, pub = creds()
    try:
        return Api(email=email, password=password, publication_url=pub, timeout=60)
    except Exception as exc:  # noqa: BLE001
        if "captcha" in str(exc).lower() or "403" in str(exc):
            die(
                f"Substack login failed: {exc}. Substack gates password logins "
                "behind a captcha. The reliable path is a session cookie: log "
                "into dspn.substack.com in Chrome, then extract cookies to "
                f"{DEFAULT_COOKIES_PATH} (see the SKILL.md cookie extraction "
                "recipe)."
            )
        die(
            f"Substack login failed: {exc}. Log into the publication once in a "
            "browser, then retry, or set up the cookies file (SKILL.md)."
        )


def pub_url() -> str:
    return os.environ.get("SUBSTACK_PUBLICATION_URL", DEFAULT_PUBLICATION)


def parse_when(iso: str) -> datetime:
    cleaned = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        die(f"could not parse datetime '{iso}': {exc}")
    if dt.tzinfo is None:
        die(
            f"schedule time must carry a UTC offset (e.g. 2026-09-24T11:00:00Z); "
            f"got naive datetime '{iso}'"
        )
    return dt.astimezone(timezone.utc)


def extract_slugs(text: str) -> list[str]:
    return sorted({match.group(4) for match in BLOG_URL_RE.finditer(text or "")})


def _posts(data) -> list:
    """Unwrap a Substack list response (the key is 'posts' for draft filters)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("posts") or data.get("drafts") or data.get("items") or []
    return []


def cmd_whoami(_args) -> None:
    api = client()
    info = api.get_user_primary_publication() or {}
    print(
        json.dumps(
            {
                "user_id": api.get_user_id(),
                "publication_name": info.get("name"),
                "publication_url": info.get("base_url")
                or info.get("custom_domain_optional")
                or info.get("hostname"),
                "handle": (info.get("author") or {}).get("handle")
                or info.get("handle"),
                "subscribers": info.get("subscriberCount")
                or info.get("subscriber_count"),
            },
            indent=2,
            default=str,
        )
    )


def cmd_drafts(args) -> None:
    """List SCHEDULED and true-draft posts via the API filters.

    The unfiltered /drafts endpoint returns the whole post archive oldest
    first (published items included), which is noise for the sync; the
    scheduled/draft filters return exactly what matters. limit must be <= 20.
    """
    api = client()
    now = datetime.now(timezone.utc)

    def shape(d: dict) -> dict:
        item = {
            "id": d.get("id"),
            "title": d.get("draft_title") or d.get("title"),
            "subtitle": d.get("draft_subtitle") or d.get("subtitle"),
            "type": d.get("type"),
            "scheduled_at": d.get("scheduled_at"),
            "post_date": d.get("post_date"),
            "should_send_email": d.get("should_send_email"),
            "is_published": d.get("is_published"),
            "blog_slugs": [],
        }
        sa = d.get("scheduled_at")
        if sa:
            item["is_future"] = datetime.fromisoformat(sa.replace("Z", "+00:00")) > now
        return item

    out: dict = {"scheduled": [], "drafts": []}
    for d in _posts(api.get_drafts(filter="scheduled", limit=args.limit)):
        item = shape(d)
        # list payloads omit draft_body; fetch each scheduled draft for slug extraction
        try:
            full = api.get_draft(d.get("id"))
            item["blog_slugs"] = extract_slugs(full.get("draft_body") or "")
        except Exception as exc:  # noqa: BLE001
            item["body_error"] = str(exc)[:160]
        out["scheduled"].append(item)
    for d in _posts(api.get_drafts(filter="draft", limit=args.limit)):
        out["drafts"].append(shape(d))
    print(json.dumps(out, indent=2, default=str))


def cmd_published(args) -> None:
    api = client()
    data = api.get_published_posts(limit=args.limit)
    items: list = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("posts") or data.get("items") or []
    pub = pub_url().rstrip("/")
    out = []
    for p in items:
        slug = p.get("slug")
        out.append(
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "slug": slug,
                "url": f"{pub}/p/{slug}" if slug else None,
                "post_date": p.get("post_date"),
                "audience": p.get("audience"),
            }
        )
    print(json.dumps({"count": len(out), "posts": out}, indent=2, default=str))


def guard_em_dashes(label: str, text: str) -> None:
    if "\u2014" in text:
        die(
            f"em dash found in {label}; scrub it (use a comma, period, or a "
            "separate sentence) and retry"
        )


def cmd_create(args) -> None:
    body_path = Path(args.body_file)
    if not body_path.exists():
        die(f"body file not found: {body_path}")
    body = body_path.read_text()
    if not body.strip():
        die(f"body file is empty: {body_path}")
    guard_em_dashes("title", args.title)
    guard_em_dashes("subtitle", args.subtitle or "")
    guard_em_dashes("body", body)
    if args.schedule_at:
        when = parse_when(args.schedule_at)

    plan = {
        "action": "create_draft",
        "title": args.title,
        "subtitle": args.subtitle or "",
        "body_chars": len(body),
        "body_words": len(body.split()),
        "banner_first_line": body.lstrip().startswith("!["),
        "blog_slugs": extract_slugs(body),
        "schedule_at": when.isoformat() if args.schedule_at else None,
        "execute": bool(args.yes),
    }
    if not args.yes:
        plan["note"] = "DRY-RUN. Re-run with --yes to create the draft."
        print(json.dumps(plan, indent=2))
        return

    api = client()
    kwargs = {"slug": args.slug} if args.slug else {}
    resp = api.create_draft_from_markdown(
        title=args.title,
        markdown=body,
        subtitle=args.subtitle or "",
        publish=False,
        **kwargs,
    )
    # newer API wraps the payload under a 'draft' key
    draft = (
        resp.get("draft")
        if isinstance(resp, dict) and isinstance(resp.get("draft"), dict)
        else resp
    )
    draft_id = draft.get("id") if isinstance(draft, dict) else None
    if not draft_id:
        die(f"draft create returned no id: {str(resp)[:300]}")
    result = {
        "draft_id": draft_id,
        "edit_url": f"{pub_url().rstrip('/')}/publish/post/{draft_id}",
        "title": args.title,
    }
    if args.schedule_at:
        result["scheduled_at"] = when.isoformat()
        result["schedule_response"] = api.schedule_draft(draft_id, when)
    print(json.dumps(result, indent=2, default=str))


def cmd_schedule(args) -> None:
    when = parse_when(args.at)
    if not args.yes:
        print(
            json.dumps(
                {
                    "action": "schedule_draft",
                    "draft_id": args.draft_id,
                    "schedule_at": when.isoformat(),
                    "note": "DRY-RUN. Re-run with --yes to schedule.",
                },
                indent=2,
            )
        )
        return
    api = client()
    print(
        json.dumps(
            {
                "draft_id": args.draft_id,
                "scheduled_at": when.isoformat(),
                "response": api.schedule_draft(args.draft_id, when),
            },
            indent=2,
            default=str,
        )
    )


def cmd_unschedule(args) -> None:
    if not args.yes:
        print(
            json.dumps(
                {
                    "action": "unschedule_draft",
                    "draft_id": args.draft_id,
                    "note": "DRY-RUN. Re-run with --yes to unschedule.",
                },
                indent=2,
            )
        )
        return
    api = client()
    print(
        json.dumps(
            {
                "draft_id": args.draft_id,
                "response": api.unschedule_draft(args.draft_id),
            },
            indent=2,
            default=str,
        )
    )


def cmd_delete(args) -> None:
    if not args.yes:
        print(
            json.dumps(
                {
                    "action": "delete_draft",
                    "draft_id": args.draft_id,
                    "note": "DRY-RUN. Re-run with --yes to permanently delete.",
                },
                indent=2,
            )
        )
        return
    api = client()
    print(
        json.dumps(
            {"draft_id": args.draft_id, "response": api.delete_draft(args.draft_id)},
            indent=2,
            default=str,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami").set_defaults(func=cmd_whoami)

    p = sub.add_parser("drafts")
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max items per filter (Substack API caps this at 20)",
    )
    p.set_defaults(func=cmd_drafts)

    p = sub.add_parser("published")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_published)

    p = sub.add_parser("create")
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", default="")
    p.add_argument("--body-file", required=True)
    p.add_argument("--slug")
    p.add_argument("--schedule-at", dest="schedule_at")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("schedule")
    p.add_argument("--draft-id", required=True)
    p.add_argument("--at", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("unschedule")
    p.add_argument("--draft-id", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_unschedule)

    p = sub.add_parser("delete")
    p.add_argument("--draft-id", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
