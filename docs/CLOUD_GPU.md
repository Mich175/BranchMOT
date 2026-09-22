# BranchMOT cloud GPU workflow

The GitHub repository is the source of truth for code. Large datasets,
checkpoints, caches, and experiment outputs live on persistent cloud storage
and must never be committed to Git.

## Recommended storage layout

Choose a persistent storage root supplied by the provider. Examples:

- AutoDL data disk: `/root/autodl-tmp/branchmot`
- RunPod network volume: `/workspace/branchmot`
- OpenBayes output volume: `/openbayes/home/branchmot`

Create this layout once:

```text
branchmot/
|-- code/          # Git clone of Mich175/BranchMOT
|-- datasets/
|   |-- DanceTrack/
|   `-- MOT17/
|-- checkpoints/
|   `-- motip/
|-- outputs/
`-- logs/
```

Only `code/` is synchronized through GitHub. The remaining directories stay
on the provider's persistent storage.

## Access the private repository

Do not paste a GitHub password or token into a notebook or commit one to the
repository. The preferred method for an ephemeral GPU machine is an SSH deploy
key with read-only repository access:

1. On the GPU instance, generate a dedicated key with `ssh-keygen -t ed25519`.
2. Copy only the public key (`~/.ssh/id_ed25519.pub`).
3. In GitHub, open **BranchMOT -> Settings -> Deploy keys -> Add deploy key**.
4. Leave **Allow write access** disabled.
5. Test with `ssh -T git@github.com`.

Then clone the code into persistent storage:

```bash
export BRANCHMOT_ROOT=/workspace/branchmot  # change for the provider
mkdir -p "$BRANCHMOT_ROOT"/{datasets,checkpoints,outputs,logs}
git clone git@github.com:Mich175/BranchMOT.git "$BRANCHMOT_ROOT/code"
cd "$BRANCHMOT_ROOT/code"
```

For a trusted long-lived machine, GitHub CLI device login is also acceptable.
Never place a personal access token directly in a clone URL because it may be
saved in shell history.

## Install and verify

Start from a recent PyTorch/CUDA image supplied by the GPU provider. Inside the
cloned repository:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
nvidia-smi
```

Clone MOTIP beside BranchMOT rather than modifying its upstream history:

```bash
cd "$BRANCHMOT_ROOT"
git clone https://github.com/MCG-NJU/MOTIP.git motip
```

Keep the exact MOTIP commit SHA in every experiment record.

## Dataset placement

If the provider has a public dataset copy, copy or bind it directly into the
persistent `datasets/` directory. Otherwise download it once from the official
source on the cloud instance. The transfer never passes through the local PC.

Expected paths:

```text
$BRANCHMOT_ROOT/datasets/DanceTrack/train
$BRANCHMOT_ROOT/datasets/DanceTrack/val
$BRANCHMOT_ROOT/datasets/DanceTrack/test
$BRANCHMOT_ROOT/datasets/MOT17/train
$BRANCHMOT_ROOT/datasets/MOT17/test
```

Before real inference, run the preflight check:

```bash
branchmot-preflight \
  --motip-root "$BRANCHMOT_ROOT/motip" \
  --data-root "$BRANCHMOT_ROOT/datasets" \
  --checkpoint "$BRANCHMOT_ROOT/checkpoints/motip/MODEL.pth"
```

## Daily workflow

At the start of a session:

```bash
cd "$BRANCHMOT_ROOT/code"
git pull --ff-only
```

Run experiments with outputs directed to `$BRANCHMOT_ROOT/outputs`, then stop
the GPU. Code changes should be committed and pushed to a `codex/*` or feature
branch. Small result summaries, configuration files, and plots may be checked
in; datasets, checkpoints, raw caches, and large logs must remain off GitHub.

## Before deleting an instance

Confirm that:

- code changes have been pushed to GitHub;
- checkpoints and results are on persistent storage;
- the provider marks that storage as retained after instance deletion;
- no secret, token, or private SSH key was copied into the repository.


