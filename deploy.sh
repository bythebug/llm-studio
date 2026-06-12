#!/usr/bin/env bash
# AWS deployment: build image → push to ECR → deploy to ECS
# Usage: ./deploy.sh [--gpu]
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — override with environment variables
# ---------------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
ECR_REPO="${ECR_REPO:-llm-studio}"
ECS_CLUSTER="${ECS_CLUSTER:-llm-studio-cluster}"
ECS_SERVICE="${ECS_SERVICE:-llm-studio-service}"
ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-llm-studio-task}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
GPU_SUPPORT="${1:-}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
confirm() {
  read -rp "$1 [y/N] " ans
  [[ "${ans,,}" == "y" ]] || die "Aborted."
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
log "Pre-flight checks"
command -v aws    >/dev/null 2>&1 || die "aws CLI not found"
command -v docker >/dev/null 2>&1 || die "docker not found"

aws sts get-caller-identity --query "Account" --output text >/dev/null \
  || die "AWS credentials not configured (run: aws configure)"

log "Deploying image tag: ${IMAGE_TAG}"
log "Target: ${ECR_URI}:${IMAGE_TAG}"
[[ "${GPU_SUPPORT}" == "--gpu" ]] && log "GPU support: enabled"
confirm "Proceed with deployment?"

# ---------------------------------------------------------------------------
# Run tests before deploying
# ---------------------------------------------------------------------------
log "Running test suite"
if command -v python3 >/dev/null 2>&1 && [ -d "venv" ]; then
  venv/bin/pytest tests/ -q || die "Tests failed — aborting deploy"
else
  log "WARNING: venv not found, skipping tests"
fi

# ---------------------------------------------------------------------------
# Build Docker image
# ---------------------------------------------------------------------------
log "Building Docker image"
DOCKERFILE="Dockerfile"
if [[ "${GPU_SUPPORT}" == "--gpu" ]]; then
  DOCKERFILE="Dockerfile.gpu"
  [ -f "${DOCKERFILE}" ] || die "${DOCKERFILE} not found. Create a GPU Dockerfile first."
fi

docker build \
  --file "${DOCKERFILE}" \
  --tag "${ECR_URI}:${IMAGE_TAG}" \
  --tag "${ECR_URI}:latest" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg GIT_COMMIT="${IMAGE_TAG}" \
  .

# ---------------------------------------------------------------------------
# Push to ECR
# ---------------------------------------------------------------------------
log "Authenticating with ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

log "Creating ECR repository if it doesn't exist"
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}"

log "Pushing image to ECR"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

# ---------------------------------------------------------------------------
# Update ECS task definition
# ---------------------------------------------------------------------------
log "Fetching current task definition"
TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition "${ECS_TASK_FAMILY}" \
  --region "${AWS_REGION}" \
  --query "taskDefinition" \
  --output json)

# Swap in the new image URI
NEW_TASK_DEF=$(echo "${TASK_DEF}" \
  | python3 -c "
import sys, json
td = json.load(sys.stdin)
for c in td['containerDefinitions']:
    if c['name'] == 'app':
        c['image'] = '${ECR_URI}:${IMAGE_TAG}'
for k in ['taskDefinitionArn','revision','status','requiresAttributes',
          'compatibilities','registeredAt','registeredBy']:
    td.pop(k, None)
print(json.dumps(td))
")

log "Registering new task definition"
NEW_TASK_ARN=$(aws ecs register-task-definition \
  --cli-input-json "${NEW_TASK_DEF}" \
  --region "${AWS_REGION}" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)
log "New task definition: ${NEW_TASK_ARN}"

# ---------------------------------------------------------------------------
# Deploy to ECS service
# ---------------------------------------------------------------------------
log "Updating ECS service"
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${NEW_TASK_ARN}" \
  --force-new-deployment \
  --region "${AWS_REGION}" \
  --output text --query "service.serviceName"

log "Waiting for deployment to stabilise (this may take a few minutes)"
aws ecs wait services-stable \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region "${AWS_REGION}"

log "Deployment complete: ${ECR_URI}:${IMAGE_TAG}"
