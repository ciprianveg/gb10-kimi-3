#!/usr/bin/env bash
# drop-caches — clear FS page cache every 60s during weight load
#
# On GB10 UMA, the NFS shard stream fills the page cache, which counts
# against the same 121 GiB unified pool as CUDA. This cron clears the
# FS page cache every 60s to prevent NVRM NV_ERR_NO_MEMORY.
set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/* && \
    echo '*/1 * * * * root sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null' > /etc/cron.d/drop-caches && \
    chmod 644 /etc/cron.d/drop-caches && \
    echo "drop-caches cron installed"
