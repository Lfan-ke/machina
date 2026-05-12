use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use machina_hw_loongarch::virt_machine::VIRT_CPU_COUNT_MAX;

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
fn smp_zero_is_rejected() {
    ensure_machina_built();

    let output = Command::new(machina_bin())
        .args(["-smp", "0"])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina should reject -smp 0; got success exit",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-smp: CPU count must be greater than 0"),
        "expected CPU-count validation error in stderr; got: {stderr}",
    );
}

#[test]
fn loongarch_smp_above_machine_limit_is_rejected() {
    ensure_machina_built();

    let too_many_cpus = (VIRT_CPU_COUNT_MAX + 1).to_string();
    let output = Command::new(machina_bin())
        .args(["-M", "loongarch64-ref", "-smp", &too_many_cpus])
        .output()
        .expect("failed to spawn machina");

    assert!(
        !output.status.success(),
        "machina should reject too many LoongArch CPUs; got success exit",
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!(
            "loongarch64-ref supports at most {VIRT_CPU_COUNT_MAX} CPUs"
        )),
        "expected LoongArch CPU-count limit in stderr; got: {stderr}",
    );
}
