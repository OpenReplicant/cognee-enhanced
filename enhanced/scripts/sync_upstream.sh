#!/usr/bin/env bash
# Merge upstream cognee into the current branch and keep its CI parked.
#
#   enhanced/scripts/sync_upstream.sh [upstream-ref]   (default: upstream/main)
#
# GitHub runs every file in .github/workflows/ (and dependabot.yml), including
# upstream's scheduled jobs, which need upstream's secrets and act on PRs. This
# fork keeps them under .github/upstream-parked/ instead. Git's rename detection
# applies upstream edits to the parked copies during the merge; this script then
# parks any workflow upstream *added* (it lands in .github/workflows/) and stops
# for manual resolution if real conflicts remain. Fork workflows are named
# enhanced_*.yml and are never moved.
set -euo pipefail

ref="${1:-upstream/main}"
parked=".github/upstream-parked"

if [[ "$ref" == upstream/* ]]; then
  git fetch upstream "${ref#upstream/}" --tags
fi

git merge --no-ff --no-commit "$ref" || true

shopt -s nullglob
for f in .github/workflows/*; do
  name="$(basename "$f")"
  [[ "$name" == enhanced_* ]] && continue
  mkdir -p "$parked/workflows"
  if [[ -e "$parked/workflows/$name" ]]; then
    # Upstream re-created a file we parked: its version replaces the parked copy.
    git rm -q --cached "$parked/workflows/$name" || true
    rm -f "$parked/workflows/$name"
  fi
  git mv -f "$f" "$parked/workflows/$name"
  echo "parked $name"
done
if [[ -e .github/dependabot.yml ]]; then
  git mv -f .github/dependabot.yml "$parked/dependabot.yml"
  echo "parked dependabot.yml"
fi

conflicts="$(git diff --name-only --diff-filter=U)"
if [[ -n "$conflicts" ]]; then
  echo "Merge has conflicts to resolve by hand, then 'git commit':" >&2
  echo "$conflicts" >&2
  exit 1
fi

if git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  git commit --no-edit
  echo "Merged $ref with upstream CI parked."
else
  echo "Nothing to merge from $ref."
fi
