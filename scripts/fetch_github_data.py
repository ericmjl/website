# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests",
# ]
# ///
"""Fetch GitHub data for the Open Source page.

This script fetches:
1. Stats for maintained projects (stars, forks, language, last update)
2. Recent contributions to other projects (PRs)

Note: Recent activity is now fetched live on the client side.

Usage:
    pixi run python scripts/fetch_github_data.py

Set GITHUB_TOKEN environment variable for higher rate limits.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

GITHUB_USERNAME = "ericmjl"
GITHUB_API = "https://api.github.com"

# Projects I maintain (will fetch live stats)
MAINTAINED_PROJECTS = [
    "pyjanitor-devs/pyjanitor",
    "ericmjl/nxviz",
    "ericmjl/llamabot",
    "ericmjl/network-analysis-made-simple",
    "ericmjl/essays-on-data-science",
]


def get_headers():
    """Get headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": GITHUB_USERNAME,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_repo_stats(repo: str) -> dict | None:
    """Fetch statistics for a repository."""
    url = f"{GITHUB_API}/repos/{repo}"
    response = requests.get(url, headers=get_headers())

    if response.status_code != 200:
        print(f"  Warning: Could not fetch {repo}: {response.status_code}")
        return None

    data = response.json()
    return {
        "name": data["name"],
        "full_name": data["full_name"],
        "description": data["description"],
        "url": data["html_url"],
        "stars": data["stargazers_count"],
        "forks": data["forks_count"],
        "open_issues": data["open_issues_count"],
        "language": data["language"],
        "updated_at": data["updated_at"],
        "pushed_at": data["pushed_at"],
        "topics": data.get("topics", []),
    }


def fetch_user_prs(username: str, days: int = 365) -> list[dict]:
    """Fetch recent PRs by user to other repositories."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"author:{username} type:pr created:>{since}"
    url = f"{GITHUB_API}/search/issues"
    params = {"q": query, "sort": "created", "order": "desc", "per_page": 100}

    response = requests.get(url, headers=get_headers(), params=params)

    if response.status_code != 200:
        print(f"  Warning: Could not fetch PRs: {response.status_code}")
        return []

    data = response.json()
    prs = []

    for item in data.get("items", []):
        # Extract repo from URL
        repo_url = item["repository_url"]
        repo_name = "/".join(repo_url.split("/")[-2:])

        # Skip PRs to own repos (we want contributions to others)
        if repo_name.startswith(f"{username}/"):
            continue

        prs.append(
            {
                "title": item["title"],
                "url": item["html_url"],
                "repo": repo_name,
                "state": item["state"],
                "created_at": item["created_at"],
                "merged": item.get("pull_request", {}).get("merged_at") is not None
                if "pull_request" in item
                else False,
            }
        )

    return prs


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "assets" / "static" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching GitHub data...")

    # Check rate limit
    rate_response = requests.get(f"{GITHUB_API}/rate_limit", headers=get_headers())
    if rate_response.status_code == 200:
        rate_data = rate_response.json()
        remaining = rate_data["resources"]["core"]["remaining"]
        print(f"  API rate limit remaining: {remaining}")
        if remaining < 50:
            print("  Warning: Low rate limit, some requests may fail")

    # 1. Fetch maintained project stats
    print("\n1. Fetching maintained project stats...")
    maintained = []
    for repo in MAINTAINED_PROJECTS:
        print(f"  Fetching {repo}...")
        stats = fetch_repo_stats(repo)
        if stats:
            maintained.append(stats)

    # Sort by stars
    maintained.sort(key=lambda x: x["stars"], reverse=True)

    # 2. Fetch recent PRs to other repos
    print("\n2. Fetching recent PRs...")
    prs = fetch_user_prs(GITHUB_USERNAME, days=365)
    print(f"  Found {len(prs)} PRs to other repos in the last year")

    # Group PRs by repo
    pr_by_repo = {}
    for pr in prs:
        repo = pr["repo"]
        if repo not in pr_by_repo:
            pr_by_repo[repo] = {"repo": repo, "prs": [], "merged_count": 0}
        pr_by_repo[repo]["prs"].append(pr)
        if pr.get("merged"):
            pr_by_repo[repo]["merged_count"] += 1

    contribution_summary = list(pr_by_repo.values())
    contribution_summary.sort(key=lambda x: len(x["prs"]), reverse=True)

    # Compile all data
    # Note: recent_activity is now fetched live on the client side
    github_data = {
        "generated_at": datetime.now().isoformat(),
        "username": GITHUB_USERNAME,
        "maintained_projects": maintained,
        "contributions": contribution_summary[:15],  # Top 15 repos
    }

    # Write output
    output_path = output_dir / "github-data.json"
    output_path.write_text(json.dumps(github_data, indent=2), encoding="utf-8")
    print(f"\nWritten GitHub data to {output_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Maintained projects: {len(maintained)}")
    print(f"  Repos contributed to: {len(contribution_summary)}")


if __name__ == "__main__":
    main()
