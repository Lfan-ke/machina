//! Regression tests for #76: `-gdb stdio` was accepted at parse time
//! but never wired up to a real RSP transport. Now reject it during
//! argument parsing so users see the failure immediately instead of a
//! misleading "machina: gdb on stdio" line followed by silent drop.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

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
fn gdb_stdio_is_rejected_at_parse_time() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-gdb", "stdio"])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina should reject -gdb stdio; got success exit",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-gdb: stdio transport is not implemented"),
        "expected stdio-not-implemented error in stderr; got: {stderr}",
    );
    assert!(
        !stderr.contains("machina: gdb on stdio"),
        "must not announce a gdb server that never starts; got: {stderr}",
    );
    assert!(
        !stderr.contains("panicked at"),
        "expected friendly error, not panic; got: {stderr}",
    );
}

#[test]
fn gdb_unsupported_transport_is_rejected() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-gdb", "pipe:nope"])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina should reject unknown -gdb transports",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-gdb: unsupported"),
        "expected unsupported-transport error; got: {stderr}",
    );
}
