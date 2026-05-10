#!/usr/bin/env bash
#
# Boot a real Linux kernel under Machina and assert userspace
# reaches the buildroot init overlay's marker line. Used by
# .github/workflows/linux-smoke-tests.yml to give Linux 6.12+
# a regression CI on the riscv64-ref machine.
#
# Resolves gevico/machina#54.
#
# Usage:
#   ./scripts/run-linux-smoke.sh
#   BR_IMG=path/to/buildroot/output/images ./scripts/run-linux-smoke.sh
#
# Required inputs:
#   ${BR_IMG}/Image       — Linux kernel image
#   ${BR_IMG}/rootfs.cpio — buildroot initramfs whose
#                            etc/init.d/S99machina-smoke prints
#                            "MACHINA_LINUX_SMOKE_OK" then poweroffs

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MACHINA_BIN="${MACHINA_BIN:-${REPO_ROOT}/target/release/machina}"
if [ ! -x "${MACHINA_BIN}" ] && [ -x "${MACHINA_BIN}.exe" ]; then
    MACHINA_BIN="${MACHINA_BIN}.exe"
fi
BR_IMG="${BR_IMG:-${REPO_ROOT}/buildroot-output/images}"
LOG_DIR="${REPO_ROOT}/target/linux-smoke"
LOG_FILE="${LOG_DIR}/run.log"
TIMEOUT_S="${TIMEOUT_S:-180}"
MARKER="MACHINA_LINUX_SMOKE_OK"

mkdir -p "${LOG_DIR}"

if [ ! -x "${MACHINA_BIN}" ]; then
    echo "error: machina binary not found: ${MACHINA_BIN}" >&2
    echo "       run: cargo build --release -p machina-emu" >&2
    exit 2
fi
if [ ! -f "${BR_IMG}/Image" ]; then
    echo "error: ${BR_IMG}/Image missing" >&2
    exit 2
fi
if [ ! -f "${BR_IMG}/rootfs.cpio" ]; then
    echo "error: ${BR_IMG}/rootfs.cpio missing" >&2
    exit 2
fi

# `|| true` swallows the inevitable non-zero exit when the kernel
# powers off via SBI reset; the actual pass/fail signal comes from
# the marker grep.
(
    timeout "${TIMEOUT_S}" "${MACHINA_BIN}" \
        -M riscv64-ref -m 256 -nographic \
        -kernel "${BR_IMG}/Image" \
        -initrd "${BR_IMG}/rootfs.cpio" \
        -append "earlycon=ns16550a,mmio,0x10000000 console=ttyS0 \
root=/dev/ram rdinit=/sbin/init" \
        || true
) | tee "${LOG_FILE}"

if grep -q "${MARKER}" "${LOG_FILE}"; then
    echo "linux-smoke: PASS"
    exit 0
fi

echo "linux-smoke: FAIL — marker '${MARKER}' not found in ${LOG_FILE}" >&2
echo "--- last 200 lines of log ---" >&2
tail -200 "${LOG_FILE}" >&2
exit 1
