#!/usr/bin/env bash
# Install the bot as a systemd service. Idempotent; safe to re-run after a pull.
#
#     sudo ./deploy/install.sh
#
# It does NOT start the bot and it does NOT write your keys. Both are on you,
# deliberately: starting a trading bot should be a decision, not a side effect.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/spreadbot}
STATE_DIR=/var/lib/spreadbot
LOG_DIR=/var/log/spreadbot
ENV_FILE=/etc/spreadbot.env
SERVICE=/etc/systemd/system/spreadbot.service
SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ $EUID -ne 0 ]]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

id -u spreadbot &>/dev/null || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin spreadbot

mkdir -p "$APP_DIR" "$STATE_DIR" "$LOG_DIR"
if [[ "$SRC_DIR" != "$APP_DIR" ]]; then
    echo "==> copying $SRC_DIR -> $APP_DIR"
    rsync -a --delete \
        --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
        --exclude 'config/config.yaml' \
        "$SRC_DIR/" "$APP_DIR/"
fi

echo "==> python environment"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -e "$APP_DIR[live]"

if [[ ! -f "$ENV_FILE" ]]; then
    install -m 600 -o root -g root "$SRC_DIR/deploy/spreadbot.env.example" "$ENV_FILE"
    echo "==> created $ENV_FILE (empty) - put your API keys there"
else
    chmod 600 "$ENV_FILE"
    chown root:root "$ENV_FILE"
fi

chown -R spreadbot:spreadbot "$APP_DIR" "$STATE_DIR" "$LOG_DIR"
# The config may hold account indices; the keys are elsewhere, but keep it tight.
[[ -f "$APP_DIR/config/config.yaml" ]] && chmod 640 "$APP_DIR/config/config.yaml"

install -m 644 "$SRC_DIR/deploy/spreadbot.service" "$SERVICE"
systemctl daemon-reload

cat <<EOF

installed. before starting:

  1. put your API keys in $ENV_FILE   (chmod 600, root only)
  2. create $APP_DIR/config/config.yaml from config/config.example.yaml
  3. check it:      sudo -u spreadbot $APP_DIR/.venv/bin/spreadbot preflight -c $APP_DIR/config/config.yaml
  4. measure first: sudo -u spreadbot $APP_DIR/.venv/bin/spreadbot measure -c $APP_DIR/config/config.yaml --duration 86400 --out $STATE_DIR/day1.jsonl

start trading only once the measurement says there is an edge:

     systemctl enable --now spreadbot
     journalctl -u spreadbot -f

kill switch (stops opening anything new, leaves existing positions managed):

     touch $STATE_DIR/HALT

EOF
