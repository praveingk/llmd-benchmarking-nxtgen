# 1.1 / granite4-7b-tiny

Experiment `1.1` saturation-observation setup and results for the Granite tiny model.

## Step-1: Deployment

- Model: `ibm-granite/granite-4.0-h-tiny`
- Topology: `4 replicas / TP=1`
- Target service: `granite4-7b-tiny-nvidia-r4-tp1`

Deployment manifest:
- `llm-inference-service-granite4-7b-tiny-baseline.yaml`


```
oc apply -f llm-inference-service-granite4-7b-tiny-baseline.yaml

```


## Saturation Benchmark

Benchmark manifest:
- `guidellm-benchmark-job-1-1-r4-tp1-concurrent-saturation.yaml`

Key runtime choices:
- Profile: `concurrent`
- Rates: `100,200,300,400`
- Workload: synthetic multi-turn with `turns=3`
- Prompt / output size: `10000 / 1000`
- Constraints: `max-seconds=300`, `max-requests=20000`, `max-errors=1000`

```
oc apply -f guidellm-benchmark-job-1-1-r4-tp1-concurrent-saturation.yaml

```



## Result Artifact

Tracked result artifact:
- `1-1-granite4-7b-tiny-r4-tp1-concurrent-saturation.csv`
- `result.log` 

```



The CSV contains the per-rate benchmark summary used for the final saturation analysis.

## Monitoring

Progress monitor:
- `../../benchmarking/monitor_guidellm_progress.py`

This helper was used to verify that a running benchmark was actively serving requests rather than hanging.
