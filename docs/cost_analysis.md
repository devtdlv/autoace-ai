# Cost Analysis

Reports $/audio-minute for the deployed local pipeline (server compute
time only — there is no per-call third-party API cost, since nothing
leaves this infrastructure), against the trial's $0.003/audio-minute
ceiling.

## Assumption stated explicitly

This repo has no cloud billing account attached, so there is no measured
invoice to report against. The figure below is derived from a stated,
clearly-labeled assumption, not a measured bill:

> **Assumption**: a 2 vCPU / 4GB general-purpose cloud VM (the class this
> service is sized for and deployed on) costs approximately **$30/month**
> — representative of budget-tier VMs in this class (e.g. DigitalOcean/
> Linode/Vultr's 2 vCPU/4GB droplets), run 24/7.

If the real deployment target uses a different provider/instance size,
substitute that monthly figure below — the $/audio-minute scales linearly
with it.

## Derivation

- Assumed cost: $30/month ÷ (30 days × 24h × 3600s) = **$0.00001157/compute-second**
- Measured throughput (`docs/latency_analysis.md`, warm): **68.6 compute-seconds per audio-minute processed**
- Cost per audio-minute = 68.6 × $0.00001157 ≈ **$0.00079/audio-minute**

## Result

**~$0.00079/audio-minute processed, against a $0.003/audio-minute ceiling**
— roughly 3.8x under budget at the assumed VM cost. The margin holds even
if the assumed VM cost is off by 2-3x (e.g. a pricier managed-cloud
instance at $60-90/month would still land at $0.0016-0.0024/minute,
still under the ceiling).

## What would change this

- **Batch processing throughput, not per-call latency, is what actually
  matters for this metric** — the pipeline runs well under real-time
  per clip (see `docs/latency_analysis.md`), so cost is driven by total
  vCPU-seconds consumed per audio-minute, not wall-clock responsiveness.
- Running multiple batch workers in parallel (more vCPUs) reduces wall-clock
  batch completion time but doesn't change $/audio-minute — total compute
  consumed per minute of audio is the same either way.
- The two dominant cost drivers are ASR and emotion fusion (83% of compute
  time, per the latency breakdown) — the same two stages flagged as
  optimization targets if this pipeline needs to scale to substantially
  higher call volume.
