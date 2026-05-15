#!/usr/bin/env bash
#
# Deploy / teardown the precise-prefix-cache-aware stack for
# ibm-granite/granite-4.1-8b on a 3-vendor pool: 4 H100-NVL + 8 MI325X +
# 8 Gaudi3 (when AMD is available; AMD release is gated by DEPLOY_AMD).
#
# Usage:
#   ./deploy.sh deploy            Install NVIDIA + Gaudi (AMD opt-in)
#   ./deploy.sh deploy-amd        Install AMD release (requires AMD nodes)
#   ./deploy.sh destroy           Remove all releases
#   ./deploy.sh redeploy          destroy + deploy
#   ./deploy.sh status            Pod / svc / release listing
#   ./deploy.sh test              Sanity completion via the EPP
#
# Env vars:
#   NAMESPACE          (default: llm-d-granite-mixed-kv)
#   HF_TOKEN           Required on first deploy if no existing secret
#   SOURCE_NAMESPACE   Namespace to copy llm-d-hf-token from (default: llm-d-granite-kv)
#   DEPLOY_AMD         "true" includes AMD release in deploy (default: false)

set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-d-granite-mixed-kv}"
SOURCE_NAMESPACE="${SOURCE_NAMESPACE:-llm-d-granite-kv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HELMFILE_DIR="$SCRIPT_DIR"

RN="granite-mixed"
EPP_RELEASE="precise-${RN}"
EPP_SVC="${EPP_RELEASE}-epp"
ZMQ_ALIAS_SVC="gaie-${RN}-epp"

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '\033[1;31m[ERR]\033[0m %s\n' "$*" >&2; exit 1; }

check_prereqs() {
    for cmd in kubectl helm helmfile jq curl; do
        command -v "$cmd" >/dev/null || err "missing required command: $cmd"
    done
    helm plugin list 2>/dev/null | grep -q '^diff' || {
        log "Installing helm-diff plugin..."
        helm plugin install https://github.com/databus23/helm-diff --verify=false
    }
}

ensure_namespace_and_token() {
    kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || {
        log "Creating namespace $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    }
    kubectl get secret llm-d-hf-token -n "$NAMESPACE" >/dev/null 2>&1 && return 0

    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "Creating llm-d-hf-token secret from HF_TOKEN env"
        kubectl create secret generic llm-d-hf-token -n "$NAMESPACE" \
            --from-literal=HF_TOKEN="$HF_TOKEN"
        return 0
    fi

    if kubectl get secret llm-d-hf-token -n "$SOURCE_NAMESPACE" >/dev/null 2>&1; then
        log "Copying llm-d-hf-token secret from $SOURCE_NAMESPACE"
        kubectl get secret llm-d-hf-token -n "$SOURCE_NAMESPACE" -o yaml \
            | sed "s/namespace: $SOURCE_NAMESPACE/namespace: $NAMESPACE/" \
            | kubectl apply -n "$NAMESPACE" -f -
        return 0
    fi

    err "HF_TOKEN env not set and no llm-d-hf-token secret found in $SOURCE_NAMESPACE"
}

ensure_scc_privileged() {
    # OpenShift: hostPath volumes + runAsUser:0 require privileged SCC.
    # Grant to default + each chart-created SA. Idempotent.
    log "Granting privileged SCC to ServiceAccounts in $NAMESPACE"
    for SA in default \
              "ms-${RN}-nvidia-llm-d-modelservice" \
              "ms-${RN}-amd-llm-d-modelservice" \
              "ms-${RN}-gaudi-llm-d-modelservice"; do
        oc adm policy add-scc-to-user privileged -z "$SA" -n "$NAMESPACE" >/dev/null 2>&1 || true
    done
}

patch_recreate_strategy() {
    # Default RollingUpdate maxSurge tries to spawn surge pods alongside old
    # ones, but on the gaudi node all 8 cards are taken, so they sit Pending.
    # Patch each ms- deployment to Recreate so future flag flips behave.
    for d in $(kubectl get deploy -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | awk '{print $1}'); do
        kubectl patch deploy "$d" -n "$NAMESPACE" --type=merge \
            -p '{"spec":{"strategy":{"type":"Recreate","rollingUpdate":null}}}' >/dev/null 2>&1 || true
    done
}

deploy_infra_and_gaie() {
    log "Installing infra + gaie releases"
    cd "$HELMFILE_DIR"
    RELEASE_NAME_POSTFIX=$RN helmfile -e istio -l 'name=infra-granite-mixed' apply -n "$NAMESPACE" --suppress-secrets >/dev/null
    RELEASE_NAME_POSTFIX=$RN helmfile -e istio -l 'name=gaie-granite-mixed' apply -n "$NAMESPACE" --suppress-secrets >/dev/null
}

deploy_vendor() {
    local vendor="$1"
    log "Installing ms-${RN}-${vendor} (decode pods)"
    cd "$HELMFILE_DIR"
    DEPLOY_AMD=true RELEASE_NAME_POSTFIX=$RN helmfile -e istio -l "name=ms-${RN}-${vendor}" apply -n "$NAMESPACE" --suppress-secrets >/dev/null
}

deploy_epp() {
    log "Installing $EPP_RELEASE (standalone EPP chart)"
    helm upgrade --install "$EPP_RELEASE" \
        oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone \
        --version v1.5.0 \
        -n "$NAMESPACE" \
        -f "$SCRIPT_DIR/standalone-values/values.yaml" >/dev/null
}

apply_httproute() {
    log "Applying httproute (decode-backend + decode-clusterip + InferencePool + HTTPRoute)"
    kubectl apply -f "$SCRIPT_DIR/httproute.yaml" -n "$NAMESPACE" >/dev/null
}

count_decode_pods_expected() {
    local n=4
    [[ "${DEPLOY_AMD:-false}" == "true" ]] && n=$((n + 8))
    n=$((n + 8))  # gaudi
    echo "$n"
}

wait_for_decode_pods() {
    local expected
    expected=$(count_decode_pods_expected)
    log "Waiting for $expected/$expected decode pods Ready (up to 60 min for first pull + warmup)"
    local deadline=$(( $(date +%s) + 3600 ))
    while true; do
        local ready
        ready=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null \
                | awk '$2=="1/1"' | wc -l | tr -d ' ')
        echo "  decode ready=$ready/$expected"
        [[ "$ready" == "$expected" ]] && break
        [[ $(date +%s) -gt $deadline ]] && err "timeout waiting for decode pods"
        sleep 30
    done
}

cmd_deploy() {
    check_prereqs
    ensure_namespace_and_token
    ensure_scc_privileged
    deploy_infra_and_gaie
    deploy_vendor nvidia
    deploy_vendor gaudi
    if [[ "${DEPLOY_AMD:-false}" == "true" ]]; then
        deploy_vendor amd
    else
        log "Skipping AMD (set DEPLOY_AMD=true once AMD nodes are available, or run './deploy.sh deploy-amd')"
    fi
    deploy_epp
    apply_httproute
    patch_recreate_strategy
    wait_for_decode_pods
    log "Deployment complete. Run '$0 test' to verify."
}

cmd_deploy_amd() {
    check_prereqs
    ensure_scc_privileged
    DEPLOY_AMD=true
    deploy_vendor amd
    patch_recreate_strategy
    log "AMD release applied. New pods will appear under llm-d.ai/accelerator-vendor=amd."
}

cmd_destroy() {
    log "Deleting helm releases in $NAMESPACE"
    helm uninstall "$EPP_RELEASE" -n "$NAMESPACE" 2>/dev/null || true
    (cd "$HELMFILE_DIR" && DEPLOY_AMD=true RELEASE_NAME_POSTFIX=$RN helmfile -e istio destroy -n "$NAMESPACE" 2>/dev/null) || true

    log "Deleting httproute artifacts"
    kubectl delete -f "$SCRIPT_DIR/httproute.yaml" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true

    log "Force-deleting any lingering decode pods"
    kubectl delete pods -n "$NAMESPACE" -l llm-d.ai/role=decode --force --grace-period=0 2>/dev/null || true

    log "Namespace + HF token retained. Delete namespace manually if desired:"
    echo "    kubectl delete namespace $NAMESPACE"
}

cmd_status() {
    log "Namespace: $NAMESPACE"
    kubectl get pods,svc,inferencepool -n "$NAMESPACE" 2>&1 || true
    echo
    helm list -n "$NAMESPACE" 2>&1 || true
    echo
    log "Decode pods by vendor:"
    kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode -o wide 2>&1 \
        | awk 'NR==1 || /-nvidia-|-amd-|-gaudi-/' || true
}

run_test() {
    log "Port-forwarding EPP and sending 3 sanity completions"
    pkill -f "port-forward.*$NAMESPACE" 2>/dev/null || true
    sleep 1
    kubectl port-forward -n "$NAMESPACE" "service/$EPP_SVC" 8000:8081 >/tmp/pf-mixed-$$.log 2>&1 &
    local pf=$!
    trap "kill $pf 2>/dev/null || true" EXIT
    sleep 3

    local body
    body=$(jq -n '{model:"ibm-granite/granite-4.1-8b",prompt:"What is Kubernetes?",max_tokens:20,temperature:0}')
    for i in 1 2 3; do
        printf '  call %d: ' "$i"
        curl -s -m 60 http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" -d "$body" \
            -o /dev/null -w "HTTP=%{http_code} time=%{time_total}s\n"
        sleep 2
    done
    { kill $pf; wait $pf; } 2>/dev/null || true
    trap - EXIT
}

case "${1:-}" in
    deploy)     cmd_deploy ;;
    deploy-amd) cmd_deploy_amd ;;
    destroy)    cmd_destroy ;;
    redeploy)   cmd_destroy; cmd_deploy ;;
    status)     cmd_status ;;
    test)       run_test ;;
    *)
        sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
        exit 1 ;;
esac
