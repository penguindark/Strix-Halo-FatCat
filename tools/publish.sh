#!/usr/bin/env bash
# publish.sh — mirror the development repo's HEAD to the public GitHub repo.
#
# Exports the tracked tree of HEAD (git archive) into the public clone,
# commits it as one snapshot commit, and pushes to origin/main. The full
# development history stays local; the public repo only ever sees clean
# snapshots of what is tracked at HEAD.
#
# Usage:
#   tools/publish.sh                  # sync + commit + push
#   tools/publish.sh --dry-run        # show what would change, push nothing
#   tools/publish.sh --message "..."  # custom snapshot commit message
#
# Env overrides: PUB_DIR, SSH_KEY, ALLOW_DIRTY=1 (skip the clean-tree guard)
set -euo pipefail

DEV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PUB_DIR=${PUB_DIR:-/home/user/app/Strix-Halo-FatCat}
SSH_KEY=${SSH_KEY:-$DEV_DIR/.github_key}
DRY=0; MSG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --message) MSG=${2:?"--message needs a value"}; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

cd "$DEV_DIR"
if [[ -n "$(git status --porcelain)" && ${ALLOW_DIRTY:-0} != 1 ]]; then
    echo "development tree is dirty — commit first (or ALLOW_DIRTY=1)" >&2
    exit 1
fi
[[ -d "$PUB_DIR/.git" ]] || { echo "public clone not found: $PUB_DIR" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "ssh key not found: $SSH_KEY" >&2; exit 1; }

# Guard: refuse to publish tracked paths that must never go public,
# even if someone git-adds them by mistake in the future.
BAD=$(git ls-tree -r --name-only HEAD \
      | grep -E '(^|/)\.github_key(\.bak)?$|(^|/)Manuals/strix-halo/|arxiv|\.pem$|(^|/)[^/]*\.key$' || true)
if [[ -n "$BAD" ]]; then
    echo "refusing: non-public paths tracked at HEAD:" >&2
    echo "$BAD" >&2
    exit 1
fi

HEAD_DESC=$(git log -1 --format='%h %s')
MSG=${MSG:-"sync: $HEAD_DESC"}

# Snapshot: wipe everything but .git, then export HEAD's tracked tree.
find "$PUB_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD | tar -x -C "$PUB_DIR"

cd "$PUB_DIR"
git config user.name  penguindark
git config user.email penguindark@hotmail.com
git add -A
if git diff --cached --quiet; then
    echo "public repo already in sync with $HEAD_DESC"
    exit 0
fi

if [[ $DRY == 1 ]]; then
    echo "[dry-run] would publish:"
    git diff --cached --stat | tail -5
    git checkout -q -- . && git clean -qfd
    exit 0
fi

git commit -q -m "$MSG"
GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes" git push origin main
echo "published $(git rev-parse --short HEAD) -> origin/main ($HEAD_DESC)"
