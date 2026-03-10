Below is an **agent-ready, concrete task plan (English)** for “try a VLA policy with **continuous / flow** outputs **without a real robot**”, assuming you have **8×H100**. I’m giving **two tracks**:

* **Track A (recommended): LeRobot + π₀ (flow) on LIBERO** → fastest path to “it runs + I can see success rates / videos”.
* **Track B: OpenPI’s LIBERO docker client/server eval** → closer to the openpi repo workflow (more moving parts).

Sources: LeRobot LIBERO install/eval docs ([Hugging Face][1]), LeRobot π₀ checkpoint card ([Hugging Face][2]), OpenPI LIBERO integration overview ([DeepWiki][3]), OpenPI LIBERO example readme ([GitHub][4]).

---

## Track A (Recommended): Run continuous/flow π₀ in LIBERO via LeRobot (PyTorch)

### Goal

1. Install LeRobot with LIBERO support
2. Run `lerobot-eval` with a pretrained **π₀ (flow)** checkpoint on a LIBERO suite
3. (Optional) record videos and/or run a short finetune

### 0) Machine prerequisites (agent checks)

* Ubuntu 20.04+ (or similar)
* NVIDIA driver + CUDA working (`nvidia-smi`)
* Python 3.10+ recommended (use conda/venv)
* Disk: plan for a few 10s of GB (models + assets)

### 1) Create environment

```bash
conda create -n vla_pi0 python=3.10 -y
conda activate vla_pi0
python -m pip install --upgrade pip wheel
```

### 2) Install LeRobot + LIBERO extra

LeRobot docs explicitly say to install LIBERO support via editable install with `.[libero]`. ([Hugging Face][1])

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[libero]"
```

### 3) Quick sanity check: can import + list tasks

```bash
python -c "import lerobot; print('lerobot ok')"
```

### 4) Run evaluation on a pretrained π₀ (flow) checkpoint

Use the HF checkpoint (example): `lerobot/pi0_libero_finetuned` ([Hugging Face][2])
LeRobot’s docs show the canonical `lerobot-eval` shape: ([Hugging Face][1])

Start with a small run:

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=2
```

Then scale up (you have 8×H100, so you can push batch/env parallelism):

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=16 \
  --eval.n_episodes=100
```

**Agent note:** tune `--eval.batch_size` based on GPU memory and env CPU load. For multi-GPU, the agent can run multiple independent `lerobot-eval` processes pinned to different GPUs, or check if LeRobot supports distributed eval in your version.

### 5) (Optional but recommended) Save videos / logs

* Agent should inspect LeRobot’s eval output directory (often under `outputs/` or similar; confirm in current CLI help).
* If CLI flags exist for video recording, enable them; otherwise, look for generated rollouts.

### 6) (Optional) Minimal “it learns” experiment

* Run a short finetune on a LIBERO dataset slice and compare success rate (before/after).
* The agent should locate LeRobot training docs/configs for π policies and use a tiny step budget first (e.g., 1k–5k updates).

---

## Track B: Use OpenPI’s official LIBERO docker workflow (policy server + runtime)

### Goal

Run OpenPI’s **two-container** setup: a **policy server** does inference, the **runtime** runs LIBERO and queries the server over WebSocket. ([DeepWiki][3])

### 0) Prereqs

* Docker + docker compose (v2)
* NVIDIA Container Toolkit configured (GPU in Docker)

### 1) Clone OpenPI and locate LIBERO example

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
ls examples/libero
```

OpenPI provides a LIBERO example/README and reports a π₀.₅ LIBERO checkpoint trained with `pi05_libero`. ([GitHub][4])

### 2) Follow the two-container compose workflow

Agent should open:

* `examples/libero/README.md` (exact commands may be there) ([GitHub][4])
* The DeepWiki “LIBERO benchmark” + “Docker deployment” pages for the architecture details (policy server + runtime) ([DeepWiki][3])

Typical flow (agent to confirm exact file names/compose targets):

1. Build images (policy + runtime)
2. Start policy server container (loads model checkpoint)
3. Start runtime container that executes LIBERO suite and queries the server

### 3) (If needed) Convert JAX checkpoint to PyTorch for local inference

OpenPI ships a conversion script path: `examples/convert_jax_model_to_pytorch.py`. ([GitHub][5])
Agent should:

* Identify which checkpoint format the chosen model is in (JAX vs PyTorch)
* Run the conversion script if required
* Validate a single inference call before running full LIBERO eval

---

## Deliverables the agent should produce

1. A short README of **exact commands that worked** on your cluster (Track A or B)
2. The final `lerobot-eval` (or openpi compose) command used
3. A small results summary:

   * Suite name (e.g., `libero_object`)
   * #episodes
   * success rate
   * where videos/logs are stored
4. Any fixes needed (mujoco/lib dependencies, docker GPU runtime tweaks, etc.)

---
