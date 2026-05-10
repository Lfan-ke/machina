//! Regression tests for #82: `-kernel` should fail fast with a clear
//! error when the file does not exist, instead of deferring the
//! failure to ELF load inside `machina_system`.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Spawn machina, wait up to `timeout` for it to exit, then kill if
/// still alive. Returns the captured stderr (best effort). Used by
/// tests that intentionally drive machina into a configuration where
/// it may loop forever without -kernel; we only care that the
/// missing-kernel guard does *not* fire in stderr.
fn machina_stderr_bounded(args: &[&str], timeout: Duration) -> String {
    let mut child = Command::new(machina_bin())
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn machina");

    std::thread::sleep(timeout);
    let _ = child.kill();
    let mut stderr = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    let _ = child.wait();
    stderr
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn machina_bin() -> PathBuf {
    let base = project_root().join("target").join("debug").join("machina");
    if cfg!(windows) {
        base.with_extension("exe")
    } else {
        base
    }
}

fn ensure_machina_built() {
    // Serialise concurrent builds: on Windows multiple linkers cannot
    // write the same .exe simultaneously.
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK.get_or_init(Mutex::default).lock().unwrap();

    let status = Command::new("cargo")
        .args(["build", "-p", "machina-emu"])
        .current_dir(project_root())
        .status()
        .expect("cargo build machina-emu failed");
    assert!(status.success(), "cargo build machina-emu failed");
}

#[test]
fn kernel_nonexistent_path_fails_fast() {
    ensure_machina_built();

    let bogus = "/this/path/does/not/exist/kernel.elf";
    let output = Command::new(machina_bin())
        .args(["-kernel", bogus])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina should reject a missing -kernel path; got success exit",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-kernel: file not found"),
        "expected 'file not found' message in stderr; got: {stderr}",
    );
    assert!(
        stderr.contains(bogus),
        "expected the offending path in stderr; got: {stderr}",
    );
}

#[test]
fn machine_help_lists_k230() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-M", "?"])
        .output()
        .expect("failed to spawn machina -M ?");

    assert!(output.status.success(), "machina -M ? should succeed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("k230"),
        "expected k230 in machine list; got: {stderr}",
    );
}

// ===== Issue #50: missing -kernel must error early, not hang =====

#[test]
fn missing_kernel_with_default_bios_errors_with_diagnostic() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-nographic"])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina without -kernel should exit non-zero, not enter the SBI loop",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no kernel specified"),
        "stderr should explain the missing kernel: {stderr}",
    );
    assert!(
        stderr.contains("-kernel"),
        "stderr should suggest the -kernel flag: {stderr}",
    );
}

#[test]
fn missing_kernel_with_bios_none_errors() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-nographic", "-bios", "none"])
        .output()
        .expect("failed to spawn machina");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no kernel specified"),
        "bare-metal mode without -kernel must also fail fast: {stderr}",
    );
}

#[test]
fn missing_kernel_with_bios_builtin_errors() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-nographic", "-bios", "builtin"])
        .output()
        .expect("failed to spawn machina");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no kernel specified"),
        "builtin SBI without -kernel must also fail fast: {stderr}",
    );
}

// Issue #50 follow-ups (codex P2): the missing-kernel check must
// not block other machines or loader-driven boots that legitimately
// supply their payload outside `-kernel`.

#[test]
fn missing_kernel_check_does_not_block_loongarch64_ref() {
    ensure_machina_built();

    // loongarch64-ref has its own boot path; the riscv-specific
    // SBI-hang guard must not fire for it. The process may not
    // exit cleanly on its own (no kernel, no payload), so we run
    // it under a short timeout and only assert that the
    // riscv-only "no kernel specified" message never appears.
    let stderr = machina_stderr_bounded(
        &["-M", "loongarch64-ref", "-nographic"],
        Duration::from_secs(3),
    );
    assert!(
        !stderr.contains("no kernel specified"),
        "the riscv SBI guard must not trigger for loongarch64-ref: \
         {stderr}",
    );
}

#[test]
fn missing_kernel_check_skipped_when_loader_supplies_payload() {
    ensure_machina_built();

    // A `-device loader` invocation drops the payload directly
    // into RAM; the SBI-hang guard must not reject it just because
    // -kernel is absent. With a bogus loader path machina will
    // either error from the loader or sit in the boot path, so
    // bound the run and only check that the missing-kernel guard
    // did not fire.
    let stderr = machina_stderr_bounded(
        &[
            "-nographic",
            "-bios",
            "none",
            "-device",
            "loader,file=/this/path/does/not/exist,addr=0x80200000",
        ],
        Duration::from_secs(3),
    );
    assert!(
        !stderr.contains("no kernel specified"),
        "loader-only boot must not trip the missing-kernel guard: \
         {stderr}",
    );
}
