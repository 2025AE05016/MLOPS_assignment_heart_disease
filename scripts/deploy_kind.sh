#!/usr/bin/env bash
# Create a kind cluster, deploy the API behind ingress-nginx and verify /predict.
set -euo pipefail
CLUSTER=${CLUSTER:-heartops}
IMAGE=${IMAGE:-heartops:local}
INGRESS_URL=https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/kind/deploy.yaml
EVIDENCE=artifacts/k8s
mkdir -p "$EVIDENCE"

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --config deploy/kind-config.yaml
kubectl config use-context "kind-$CLUSTER"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER"

kubectl apply -f "$INGRESS_URL"
kubectl -n ingress-nginx rollout status deployment/ingress-nginx-controller --timeout=240s
kubectl -n ingress-nginx wait --for=condition=complete job/ingress-nginx-admission-patch --timeout=120s

EXISTING=$(kubectl get deployment heartops -o name 2>/dev/null || true)
kubectl apply -f deploy/api.yaml -f deploy/ingress.yaml
# Same image tag on redeploy: restart so pods pick up the freshly loaded image.
[ -n "$EXISTING" ] && kubectl rollout restart deployment/heartops
kubectl rollout status deployment/heartops --timeout=240s
# Let pods from the previous ReplicaSet finish terminating before capturing evidence.
for i in $(seq 1 45); do
  [ "$(kubectl get pods -l app=heartops --no-headers | grep -vc ' Running ')" = 0 ] && break
  sleep 2
done
kubectl wait --for=jsonpath='{.status.loadBalancer.ingress}' ingress/heartops --timeout=120s

for i in $(seq 1 30); do curl -fsS http://localhost:8081/ready && break; sleep 2; done
echo
{
  echo "\$ kubectl get nodes -o wide"; kubectl get nodes -o wide; echo
  echo "\$ kubectl get deploy,pods,svc,ingress -l app=heartops -o wide"
  kubectl get deploy,pods,svc,ingress -l app=heartops -o wide; echo
  echo "\$ curl http://localhost:8081/health"; curl -fsS http://localhost:8081/health; echo
  echo "\$ curl -X POST http://localhost:8081/predict -d @examples/patient.json"
  curl -fsS -H 'Content-Type: application/json' --data @examples/patient.json \
    http://localhost:8081/predict; echo
} | tee "$EVIDENCE/deployment.txt"
kubectl describe deployment heartops > "$EVIDENCE/describe-deployment.txt"
kubectl logs -l app=heartops --tail=20 --prefix > "$EVIDENCE/pod-logs.txt"
echo "Evidence written to $EVIDENCE/"
