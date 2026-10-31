#!/bin/bash
# Start MyAI. Run ./install.sh first (and again to change settings).
set -u

VPY=".venv/bin/python"
ENV_FILE=".env"

# PORT from .env, ignoring an inline comment; empty if it isn't set.
port_from_env() {
  sed -n 's/^PORT=[[:space:]]*"\{0,1\}\([0-9][0-9]*\).*/\1/p' "$ENV_FILE" | tail -n 1
}

main() {
  cd "$(dirname "$0")" || exit 1

  if [ ! -x "$VPY" ] || [ ! -f "$ENV_FILE" ]; then
    echo "MyAI isn't set up yet. Run ./install.sh first."
    exit 1
  fi

  local port host status
  port=$(port_from_env)
  echo
  echo "  Starting MyAI…"
  if [ -n "$port" ]; then
    echo "    On this Mac:           http://localhost:$port"
    if host=$(scutil --get LocalHostName 2>/dev/null) && [ -n "$host" ]; then
      echo "    On your home network:  http://$host.local:$port"
    fi
  fi
  echo "  Loading the AI model can take a minute. Press Ctrl-C to stop."
  echo

  # Ctrl-C goes to the app, which shuts down; the script carries on to say so
  # (and ends the line the app leaves unfinished).
  trap ':' INT
  "$VPY" app.py
  status=$?
  trap - INT

  echo
  if [ "$status" -eq 0 ] || [ "$status" -eq 130 ]; then
    echo "  MyAI has stopped."
    return 0
  fi
  echo "  MyAI stopped with an error (code $status). See the messages above."
  return "$status"
}

# Tests source this file without starting anything.
if [ -z "${MYAI_START_NO_MAIN:-}" ]; then
  main "$@"
fi
