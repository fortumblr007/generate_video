#!/bin/bash

# Start ComfyUI in the background, then register the RunPod worker immediately.
# Hub's "prepare AI API" deadline expires if handler.py has not called
# runpod.serverless.start() yet; waiting for Comfy here causes that timeout.
set -e

# /proc/meminfo can expose host RAM instead of the worker's cgroup limit.
host_memory_kib=$(awk '/MemTotal:/ { print $2 }' /proc/meminfo)
cgroup_memory_limit="unknown"
if [ -r /sys/fs/cgroup/memory.max ]; then
    cgroup_memory_limit=$(cat /sys/fs/cgroup/memory.max)
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    cgroup_memory_limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
fi
echo "Memory limits: host_mem_total_kib=${host_memory_kib:-unknown} cgroup_limit_bytes=$cgroup_memory_limit"

# Avoid retaining many per-thread allocator arenas between warm-worker jobs.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

echo "Starting ComfyUI in the background..."
# Pinned offload buffers are charged to the container cgroup, but ComfyUI sizes
# them using the much larger host RAM total on RunPod. Prefer the worker's fast
# local disk for dynamic model loading and avoid caching one-shot node outputs.
python /ComfyUI/main.py \
    --listen \
    --use-sage-attention \
    --disable-pinned-memory \
    --fast-disk \
    --cache-none &

echo "Starting the handler..."
exec python handler.py
