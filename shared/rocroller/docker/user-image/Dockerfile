# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

ARG base_image

FROM ${base_image}

ADD docker/user-image/user-entrypoint /opt/user-entrypoint
ENTRYPOINT ["/opt/user-entrypoint"]

RUN [ -z ${HTTP_PROXY} ]  || (echo "export  HTTP_PROXY=\"${HTTP_PROXY}\";" >> /etc/skel/.bashrc)
RUN [ -z ${HTTPS_PROXY} ] || (echo "export HTTPS_PROXY=\"${HTTPS_PROXY}\";" >> /etc/skel/.bashrc)
RUN [ -z ${FTP_PROXY} ]   || (echo "export   FTP_PROXY=\"${FTP_PROXY}\";" >> /etc/skel/.bashrc)

RUN truncate -s 0 /var/log/faillog
RUN truncate -s 0 /var/log/lastlog

# Python
ARG _UID
ARG _GID
RUN chown -R ${_UID}:${_GID} /opt/rr-python-venv
