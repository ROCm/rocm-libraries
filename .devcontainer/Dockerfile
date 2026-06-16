FROM rocm/composable_kernel:ck_ub24.04_rocm7.13_therock

ARG USERNAME=dev
# UID/GID default to 1000; VS Code's updateRemoteUserUID remaps these at
# container start to match the host. For plain `docker run` use, pass
# --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g).
ARG USER_UID=1000
ARG USER_GID=1000
ARG VIDEO_GID=44
ARG RENDER_GID=109

RUN set -eux; \
    : "${USERNAME:?build arg USERNAME is required}"; \
    if getent group "$USER_GID" >/dev/null; then \
        existing_group="$(getent group "$USER_GID" | cut -d: -f1)"; \
        [ "$existing_group" = "$USERNAME" ] || groupmod -n "$USERNAME" "$existing_group"; \
    else \
        groupadd --gid "$USER_GID" "$USERNAME"; \
    fi; \
    if getent passwd "$USER_UID" >/dev/null; then \
        existing_user="$(getent passwd "$USER_UID" | cut -d: -f1)"; \
        [ "$existing_user" = "$USERNAME" ] || usermod -l "$USERNAME" -d "/home/$USERNAME" -m "$existing_user"; \
    else \
        useradd --uid "$USER_UID" --gid "$USER_GID" -m -s /bin/bash "$USERNAME"; \
    fi; \
    # Ensure render/video groups exist with host GIDs and add user to them so
    # /dev/kfd and /dev/dri/* are accessible inside the container.
    if ! getent group "$VIDEO_GID" >/dev/null; then groupadd --gid "$VIDEO_GID" hostvideo; fi; \
    if ! getent group "$RENDER_GID" >/dev/null; then groupadd --gid "$RENDER_GID" hostrender; fi; \
    usermod -aG "$VIDEO_GID","$RENDER_GID" "$USERNAME"

# sudo + a few quality-of-life tools. Node is needed for Claude Code.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sudo git curl ca-certificates gnupg openssh-client less vim \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/rocm-libraries