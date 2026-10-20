#!/usr/bin/env bash
set -e

# -----------------------------
# Load secrets
# -----------------------------
set -a
source ~/.drml_secrets
set +a

APP_DIR="$HOME/dr-ml"
API_PORT=8000
UI_PORT=8501

echo "Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip

echo "Preparing repo URL with token (used only for clone/pull)..."
AUTH_REPO_URL=$(echo "$GIT_REPO_URL" | sed "s#https://#https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@#")

echo "Cloning or updating repo..."
if [ -d "$APP_DIR/.git" ]; then
  cd "$APP_DIR"
  git fetch origin
  git checkout "${BRANCH:-main}"
  git pull origin "${BRANCH:-main}"
else
  git clone -b "${BRANCH:-main}" "$AUTH_REPO_URL" "$APP_DIR"
  cd "$APP_DIR"
fi

# Optional: remove token from stored git remote after clone
git remote set-url origin "$GIT_REPO_URL"

echo "Creating venv and installing Python deps..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Writing EC2 .env (training + inference)..."
cat > .env <<EOF
PROJECT_ROOT="$APP_DIR"

LOG_PATH="\${PROJECT_ROOT}/logs/app.log"

DATASET_DIR="\${PROJECT_ROOT}/dataset"
DIABETES_DATASET_PATH="\${DATASET_DIR}/diabetes.csv"
HEART_DISEASE_DATASET_PATH="\${DATASET_DIR}/heart.csv"

MODEL_DIR="\${PROJECT_ROOT}/model_dir"
DIABETES_MODEL_PATH="\${MODEL_DIR}/diabetes_prediction_pipeline.joblib"
HEART_DISEASE_MODEL_PATH="\${MODEL_DIR}/heart_disease_prediction_pipeline.joblib"

DIABETES_TARGET_COL="Outcome"
HEART_DISEASE_TARGET_COL="target"

HYPER_PARAMS_YAML_PATH="\${PROJECT_ROOT}/src/training/config/best_hyperparams.yaml"

TEST_SIZE=0.15
RANDOM_STATE=42

API_HOST="0.0.0.0"
API_PORT=${API_PORT}
API_URL="http://127.0.0.1:${API_PORT}/api/predict"
EOF

set -a
source .env
set +a


mkdir -p logs model_dir

echo "Stopping old processes (if any)..."
pkill -f "uvicorn src.backend.main:app" || true
pkill -f "streamlit run" || true

echo "Starting FastAPI (uvicorn)..."
nohup .venv/bin/uvicorn src.backend.main:app --host 0.0.0.0 --port ${API_PORT} > api.log 2>&1 &

echo "Starting Streamlit..."
nohup .venv/bin/streamlit run src/frontend/app.py --server.address 0.0.0.0 --server.port ${UI_PORT} > ui.log 2>&1 &

echo ""
echo "Done."
echo "API health:  curl http://127.0.0.1:${API_PORT}/api/health"
echo "Frontend:    http://<EC2_PUBLIC_IP>:${UI_PORT}"
echo "Logs:"
echo "  tail -n 200 $APP_DIR/api.log"
echo "  tail -n 200 $APP_DIR/ui.log"


