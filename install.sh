#!/usr/bin/env bash
#
# install.sh — MyAI installer launcher (macOS / Apple Silicon).
#
# Run from inside a cloned MyAI repo:
#     git clone https://github.com/TheTechy/MyAI.git
#     cd MyAI
#     ./install.sh
#
# This script only does what Python can't do for itself: find a suitable
# Python, create the venv MyAI owns (./.venv), then hand over to the setup
# app (setup_myai.py), which runs inside that venv and does everything else.
#
# Nothing here needs sudo. Everything installs into ./.venv, local to the repo.
#
# Knobs (override by exporting before running, e.g. VENV_DIR=.venv2 ./install.sh):
#   MIN_MAJOR / MIN_MINOR   minimum Python (default 3.13)
#   VENV_DIR                venv location (default .venv)

set -euo pipefail

MIN_MAJOR="${MIN_MAJOR:-3}"
MIN_MINOR="${MIN_MINOR:-13}"
VENV_DIR="${VENV_DIR:-.venv}"
SETUP_APP="setup_myai.py"
PYTHON_DOWNLOAD_PAGE="https://www.python.org/downloads/macos/"

# ── Pretty output ──────────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; DIM=""; G=""; Y=""; R=""; C=""; N=""
fi
info(){ printf '%s\n' "  $*"; }
ok(){ printf '%s\n' "  ${G}✔${N} $*"; }
warn(){ printf '%s\n' "  ${Y}⚠${N}  $*"; }
err(){ printf '%s\n' "  ${R}✘${N} $*" >&2; }
die(){ err "$*"; exit 1; }

banner(){
  printf '\n%s\n' "  ${B}${C}MyAI${N} ${DIM}· Private. Local. Yours.${N}"
  printf '%s\n\n' "  ${DIM}Getting things ready — this part takes a few seconds.${N}"
}

# ── Preflight ───────────────────────────────────────────────────────────────────
preflight(){
  [ "$(uname -s)" = "Darwin" ] || die "This installer is macOS-only for now (detected $(uname -s))."
  [ "$(uname -m)" = "arm64" ] || warn "This Mac isn't Apple Silicon — MyAI will run, but slowly (no GPU)."
  [ -f "$SETUP_APP" ] && [ -d "core" ] || die "Run this from the MyAI folder you cloned (couldn't find $SETUP_APP)."
}

# ── Python discovery (version-gated) ────────────────────────────────────────────
# Candidate interpreters, most likely first. Searches absolute paths as well as
# PATH, because a minimal PATH (e.g. launched from Finder) can hide the real one
# behind Apple's /usr/bin/python3 (3.9).
python_candidates(){
  local name p
  for name in python3.14 python3.13 python3; do
    command -v "$name" 2>/dev/null || true
  done
  ls -d /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 2>/dev/null | sort -rV || true
  for p in /opt/homebrew/bin/python3.* /usr/local/bin/python3.*; do
    case "$p" in *-config|*'*'*) ;; *) printf '%s\n' "$p" ;; esac
  done
  printf '%s\n' /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3
}

# version_ok "3.13" → 0 if it meets MIN_MAJOR.MIN_MINOR.
version_ok(){
  local maj="${1%%.*}" min="${1##*.}"
  [ "$maj" -gt "$MIN_MAJOR" ] || { [ "$maj" -eq "$MIN_MAJOR" ] && [ "$min" -ge "$MIN_MINOR" ]; }
}

# pick_python CANDIDATE... → prints the first interpreter that runs and meets
# the minimum, returns 0. On failure prints the newest version it did see
# (possibly empty) and returns 1. Prints rather than setting a global because
# callers use $(...), where an assignment wouldn't survive.
pick_python(){
  local p v best=""
  for p in "$@"; do
    [ -x "$p" ] || continue
    v="$("$p" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" || continue
    if version_ok "$v"; then printf '%s' "$p"; return 0; fi
    if [ -z "$best" ] || [ "$(printf '%s\n%s\n' "$best" "$v" | sort -V | tail -1)" = "$v" ]; then best="$v"; fi
  done
  printf '%s' "$best"; return 1
}

PYBIN=""
ensure_python(){
  local result candidates=()
  while IFS= read -r p; do [ -n "$p" ] && candidates+=("$p"); done < <(python_candidates)
  if result="$(pick_python "${candidates[@]}")"; then
    PYBIN="$result"
    ok "Python $("$PYBIN" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])') ${DIM}($PYBIN)${N}"
    return 0
  fi
  if [ -n "$result" ]; then
    err "MyAI needs Python ${MIN_MAJOR}.${MIN_MINOR} or newer — the newest on this Mac is $result."
  else
    err "MyAI needs Python ${MIN_MAJOR}.${MIN_MINOR} or newer, and none was found on this Mac."
  fi
  info "The easiest fix is the official installer from python.org:"
  info "  ${C}$PYTHON_DOWNLOAD_PAGE${N}"
  printf '\n'
  # Only open the page if someone actually answered — not when stdin is closed.
  if read -r -p "  Open that page now? [Y/n] " a; then
    case "${a:-Y}" in [Nn]*) : ;; *) open "$PYTHON_DOWNLOAD_PAGE" 2>/dev/null || true ;; esac
  fi
  info "When Python is installed, run ${B}./install.sh${N} again."
  exit 1
}

# ── venv ────────────────────────────────────────────────────────────────────────
VPY=""
ensure_venv(){
  VPY="$VENV_DIR/bin/python"
  if [ -x "$VPY" ]; then
    local v
    if v="$("$VPY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" && version_ok "$v"; then
      ok "Using the existing ${VENV_DIR} (Python $v)"
      return 0
    fi
    # Our own venv, but broken or built from a too-old Python (e.g. that Python
    # was uninstalled) — only packages live there, so rebuilding it is safe.
    warn "${VENV_DIR} is out of date or broken — rebuilding it"
    rm -rf "$VENV_DIR"
  fi
  "$PYBIN" -m venv "$VENV_DIR" || die "Couldn't create ${VENV_DIR}."
  ok "Created ${VENV_DIR}"
  "$VPY" -m pip install --quiet --disable-pip-version-check --upgrade pip \
    || warn "Couldn't upgrade pip — carrying on with the bundled one"
}

main(){
  # repo root = this script's dir, so it works however it's invoked
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  banner
  preflight
  ensure_python
  ensure_venv
  exec "$VPY" "$SETUP_APP" "$@"
}

# Allow sourcing for tests without running the install (MYAI_INSTALL_NO_MAIN=1).
[ "${MYAI_INSTALL_NO_MAIN:-0}" = "1" ] || main "$@"
