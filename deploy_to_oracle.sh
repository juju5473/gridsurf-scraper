#!/usr/bin/env bash
# deploy_to_oracle.sh — GridSurf on VM.Standard.E2.1.Micro (AMD, Always Free, Toronto)

set -uo pipefail

TENANCY="ocid1.tenancy.oc1..aaaaaaaag76b6fg3dg7563nppoji54n2gj2xbnhucvjq5jj626wtv22renqq"
SUBNET_ID="ocid1.subnet.oc1.ca-toronto-1.aaaaaaaamc5pcqiexhsfryarxr4tskqvfns6ywnkeur3whfghx5gq766o7kq"
SECLIST_ID="ocid1.securitylist.oc1.ca-toronto-1.aaaaaaaala77livluu5gdi7hs4aihigtzopibddhzfaf64onyxo3inv7xc3a"
AD="Cadz:CA-TORONTO-1-AD-1"
IMAGE_ID="ocid1.image.oc1.ca-toronto-1.aaaaaaaatz6m5qonfwyscip3xkha7pafbgd2b4srndifatfinrmuaksz2buq"
SSH_KEY_FILE="$HOME/.ssh/gridsurf_oracle"
SSH_PUB_KEY_FILE="$HOME/.ssh/gridsurf_oracle.pub"
LOCAL_SCRAPER_DIR="$HOME/Desktop/GBC/deep learning/Gridsurf/gridsurf_scraper"
LOG_FILE="$HOME/Desktop/GBC/deep learning/Gridsurf/deploy.log"

export SUPPRESS_LABEL_WARNING=True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Step 0: open port 8080 in OCI security list ────────────────────────────

log "Opening port 8080 in OCI security list..."
INGRESS_RULES='[
  {"is-stateless":false,"protocol":"6","source":"0.0.0.0/0","source-type":"CIDR_BLOCK",
   "tcp-options":{"destination-port-range":{"max":22,"min":22},"source-port-range":null}},
  {"is-stateless":false,"protocol":"6","source":"0.0.0.0/0","source-type":"CIDR_BLOCK",
   "tcp-options":{"destination-port-range":{"max":8080,"min":8080},"source-port-range":null}}
]'
oci network security-list update \
  --security-list-id "$SECLIST_ID" \
  --ingress-security-rules "$INGRESS_RULES" \
  --force 2>/dev/null \
  && log "Port 8080 opened in security list." \
  || log "WARN: Security list update failed — may need to open port 8080 manually."

# ── Step 1: launch VM.Standard.E2.1.Micro with retry ──────────────────────

log "Launching VM.Standard.E2.1.Micro (AMD, Always Free) in Toronto..."

INSTANCE_ID=""
while true; do
  TMPOUT=$(mktemp)
  oci compute instance launch \
    --compartment-id "$TENANCY" \
    --availability-domain "$AD" \
    --shape "VM.Standard.E2.1.Micro" \
    --display-name "gridsurf-scraper" \
    --image-id "$IMAGE_ID" \
    --subnet-id "$SUBNET_ID" \
    --assign-public-ip true \
    --ssh-authorized-keys-file "$SSH_PUB_KEY_FILE" \
    --boot-volume-size-in-gbs 50 \
    > "$TMPOUT" 2>&1 &
  OCI_PID=$!
  ( sleep 150; kill "$OCI_PID" 2>/dev/null ) &
  KILLER_PID=$!
  wait "$OCI_PID" 2>/dev/null
  kill "$KILLER_PID" 2>/dev/null; wait "$KILLER_PID" 2>/dev/null
  RESULT=$(cat "$TMPOUT"); rm -f "$TMPOUT"

  if echo "$RESULT" | grep -qi "out of host capacity"; then
    log "Out of capacity. Retrying in 60s..."
    sleep 60; continue
  fi

  if echo "$RESULT" | grep -q '"id":'; then
    INSTANCE_ID=$(echo "$RESULT" | grep '"id"' | head -1 | \
      sed 's/.*"id": *"\([^"]*\)".*/\1/')
    log "Instance launched: $INSTANCE_ID"
    break
  fi

  if [[ -z "$RESULT" ]]; then
    log "API call timed out. Retrying in 60s..."
  else
    log "Error: $(echo "$RESULT" | grep '"message"' | head -1). Retrying in 60s..."
  fi
  sleep 60
done

echo "$INSTANCE_ID" > /tmp/gridsurf_instance_id.txt

# ── Step 2: wait for RUNNING ───────────────────────────────────────────────

log "Waiting for instance to reach RUNNING state..."
while true; do
  STATE=$(oci compute instance get \
    --instance-id "$INSTANCE_ID" \
    2>/dev/null | grep '"lifecycle-state"' | head -1 | \
    sed 's/.*"lifecycle-state": *"\([^"]*\)".*/\1/')
  log "State: $STATE"
  [[ "$STATE" == "RUNNING" ]] && break
  sleep 10
done

# ── Step 3: get public IP ──────────────────────────────────────────────────

log "Getting public IP..."
PUBLIC_IP=""
while [[ -z "$PUBLIC_IP" ]]; do
  PUBLIC_IP=$(oci compute instance list-vnics \
    --instance-id "$INSTANCE_ID" \
    --compartment-id "$TENANCY" \
    2>/dev/null | grep '"public-ip"' | grep -v 'null' | head -1 | \
    sed 's/.*"public-ip": *"\([^"]*\)".*/\1/')
  [[ -z "$PUBLIC_IP" ]] && sleep 5
done
log "Public IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/gridsurf_public_ip.txt

# ── Step 4: wait for SSH ───────────────────────────────────────────────────

gssh() { ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" "$@"; }
gscp() { scp -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no "$@"; }

log "Waiting for SSH to become available..."
until ssh -i "$SSH_KEY_FILE" \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=5 \
      -o BatchMode=yes \
      ubuntu@"$PUBLIC_IP" "echo ok" &>/dev/null; do
  sleep 10
done
log "SSH is up."

# ── Step 5: install Python packages ───────────────────────────────────────

log "Installing Python packages on VM..."
gssh "sudo apt-get update -qq && \
  sudo apt-get install -y -qq python3 python3-pip && \
  pip3 install --quiet \
    requests schedule beautifulsoup4 python-dotenv \
    numpy pandas scikit-learn zstandard"
log "Packages installed."

# ── Step 6: copy all project files ────────────────────────────────────────

log "Creating ~/gridsurf_scraper/data on VM..."
gssh "mkdir -p ~/gridsurf_scraper/data"

log "Copying files..."
FILES=(
  scraper.py
  analyze.py
  price_model.py
  notifier.py
  backtest.py
  price_forecast.py
  dashboard_api.py
  dashboard.html
  .env
)
for f in "${FILES[@]}"; do
  if [[ -f "$LOCAL_SCRAPER_DIR/$f" ]]; then
    gscp "$LOCAL_SCRAPER_DIR/$f" ubuntu@"$PUBLIC_IP":~/gridsurf_scraper/
    log "  Copied $f"
  else
    log "  WARN: $f not found locally, skipping"
  fi
done
log "All files copied."

# ── Step 7: open port 8080 in VM host firewall ─────────────────────────────

log "Opening port 8080 in UFW..."
gssh "sudo ufw allow 8080/tcp && sudo ufw --force enable" || true

# ── Step 8: create gridsurf scraper systemd service ───────────────────────

log "Creating gridsurf.service (scraper)..."
SCRAPER_UNIT='[Unit]
Description=GridSurf Scraper
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gridsurf_scraper
EnvironmentFile=/home/ubuntu/gridsurf_scraper/.env
ExecStart=/usr/bin/python3 scraper.py --schedule
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/ubuntu/gridsurf_scraper/scraper.log
StandardError=append:/home/ubuntu/gridsurf_scraper/scraper.log

[Install]
WantedBy=multi-user.target'

echo "$SCRAPER_UNIT" | ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" \
  "sudo tee /etc/systemd/system/gridsurf.service > /dev/null"

# ── Step 9: create dashboard API systemd service ───────────────────────────

log "Creating gridsurf-api.service (dashboard)..."
API_UNIT='[Unit]
Description=GridSurf Dashboard API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gridsurf_scraper
ExecStart=/usr/bin/python3 dashboard_api.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/ubuntu/gridsurf_scraper/api.log
StandardError=append:/home/ubuntu/gridsurf_scraper/api.log

[Install]
WantedBy=multi-user.target'

echo "$API_UNIT" | ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" \
  "sudo tee /etc/systemd/system/gridsurf-api.service > /dev/null"

# ── Step 10: enable and start both services ────────────────────────────────

gssh "sudo systemctl daemon-reload && \
  sudo systemctl enable gridsurf gridsurf-api && \
  sudo systemctl start gridsurf gridsurf-api"
log "Both services enabled and started."

# ── Step 11: verify ────────────────────────────────────────────────────────

sleep 5
SCRAPER_STATUS=$(gssh "sudo systemctl is-active gridsurf")
API_STATUS=$(gssh "sudo systemctl is-active gridsurf-api")
log "gridsurf.service:     $SCRAPER_STATUS"
log "gridsurf-api.service: $API_STATUS"

log "Scraper log (last 10 lines):"
gssh "tail -n 10 ~/gridsurf_scraper/scraper.log 2>/dev/null || echo '(not yet written)'"

log "API log (last 5 lines):"
gssh "tail -n 5 ~/gridsurf_scraper/api.log 2>/dev/null || echo '(not yet written)'"

# ── Summary ────────────────────────────────────────────────────────────────

log ""
log "══════════════════════════════════════════════════════════════"
log "  Deployment complete!"
log "  Instance:     $INSTANCE_ID"
log "  Public IP:    $PUBLIC_IP"
log "  SSH:          ssh -i ~/.ssh/gridsurf_oracle ubuntu@$PUBLIC_IP"
log "  Dashboard:    http://$PUBLIC_IP:8080"
log "  API logs:     ssh in → tail -f ~/gridsurf_scraper/api.log"
log "  Scraper logs: ssh in → tail -f ~/gridsurf_scraper/scraper.log"
log "══════════════════════════════════════════════════════════════"
