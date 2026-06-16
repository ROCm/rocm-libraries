# Dev Container — rocm-libraries

Base image: `rocm/composable_kernel:ck_ub24.04_rocm7.13_therock`

## What this gives you

- ROCm 7.13 + Composable Kernel toolchain preinstalled.
- A container user matching the host's `$USER`, UID, and GID, so files created
  inside the container are owned by you on the host.
- Membership in host `video` (GID 44) and `render` (GID 109) groups for
  `/dev/kfd` and `/dev/dri/*` access.
- Claude Code CLI (`@anthropic-ai/claude-code`) preinstalled, plus the
  VS Code extension auto-installed on first attach.
- Host identity bind-mounted in:
  - `~/.claude` and `~/.claude.json` — Claude auth, settings, projects, memory.
  - `~/.gitconfig` (ro) — your git `user.name` / `user.email`.
  - `~/.ssh` (ro) — SSH keys for `git push`.

No re-login to Claude, no re-config of git inside the container.

## How the user is set up

- `USERNAME` is taken from your host `$USER` at build time via
  `${localEnv:USER}` in `devcontainer.json`.
- UID/GID default to `1000` in the `Dockerfile`. VS Code's
  `updateRemoteUserUID: true` then **remaps the user's UID/GID to your host's
  at container start**, so files you create are owned by you on the host.

No extra env vars required.

## First-run note: git identity

If you don't have `~/.gitconfig` on the host, an empty one is created
automatically (by `initializeCommand`) so the bind mount works. Set your
identity once on the host so commits inside the container are attributed:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

## Use from VS Code

1. Install the "Dev Containers" extension.
2. Open this repo, `F1` → **Dev Containers: Reopen in Container**.
3. Once attached, the Claude Code extension is available; the CLI is also on
   `$PATH` (`claude` in any terminal).

## Use from the CLI (without VS Code)

```bash
cd /path/to/rocm-libraries

docker build \
  --build-arg USERNAME="$USER" \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -t rocm-libraries-dev .devcontainer

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=16g \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v "$PWD":/workspaces/rocm-libraries \
  -v "$HOME/.claude":"/home/$USER/.claude" \
  -v "$HOME/.claude.json":"/home/$USER/.claude.json" \
  -v "$HOME/.gitconfig":"/home/$USER/.gitconfig:ro" \
  -v "$HOME/.ssh":"/home/$USER/.ssh:ro" \
  -w /workspaces/rocm-libraries \
  rocm-libraries-dev bash
```

Then run `claude` inside.