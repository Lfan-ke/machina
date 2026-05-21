// MMP: Machina Monitor Protocol (restricted QMP subset).
//
// JSON wire protocol compatible with QMP format:
// - Greeting on connect
// - {"execute":"cmd","arguments":{}} requests
// - {"return":{}} or {"error":{}} responses

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::service::MonitorService;

/// Decode `CARGO_PKG_VERSION` into a (major, minor, micro) triple
/// for the QMP-compatible `query-version` response. Non-numeric or
/// missing components default to 0.
fn parse_pkg_version() -> (u64, u64, u64) {
    let mut parts = env!("CARGO_PKG_VERSION").split('.');
    let parse = |s: Option<&str>| -> u64 {
        s.and_then(|p| p.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(0)
    };
    (
        parse(parts.next()),
        parse(parts.next()),
        parse(parts.next()),
    )
}

/// Build the QMP greeting line. The advertised `version` triple is
/// derived from the same `CARGO_PKG_VERSION` source as
/// `query-version`, so a single session never reports two
/// different versions to a client.
pub fn greeting() -> String {
    let (major, minor, micro) = parse_pkg_version();
    json!({
        "QMP": {
            "version": {
                "machina": {
                    "major": major,
                    "minor": minor,
                    "micro": micro,
                }
            },
            "capabilities": [],
        }
    })
    .to_string()
}

/// Run MMP server on a TCP listener. Blocks until
/// quit is requested.
pub fn run_tcp(listener: TcpListener, svc: Arc<Mutex<MonitorService>>) {
    listener.set_nonblocking(false).expect("set_nonblocking");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if handle_connection(s, &svc) {
                    break; // quit requested
                }
            }
            Err(_) => {
                if svc
                    .lock()
                    .expect("monitor mutex poisoned")
                    .state
                    .is_quit_requested()
                {
                    break;
                }
            }
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    svc: &Arc<Mutex<MonitorService>>,
) -> bool {
    // Send greeting.
    let _ = writeln!(stream, "{}", greeting());
    let _ = stream.flush();

    let reader =
        BufReader::new(stream.try_clone().expect("failed to clone TCP stream"));
    let mut caps_done = false;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({
                    "error": {
                        "class": "GenericError",
                        "desc": format!(
                            "JSON parse error: {}",
                            e
                        )
                    }
                });
                let _ = writeln!(stream, "{}", err);
                let _ = stream.flush();
                continue;
            }
        };

        let cmd = req["execute"].as_str().unwrap_or("").to_string();

        if !caps_done && cmd != "qmp_capabilities" {
            let err = json!({
                "error": {
                    "class": "CommandNotFound",
                    "desc":
                        "qmp_capabilities required first"
                }
            });
            let _ = writeln!(stream, "{}", err);
            let _ = stream.flush();
            continue;
        }

        let resp = dispatch(&cmd, svc);
        let _ = writeln!(stream, "{}", resp);
        let _ = stream.flush();

        if cmd == "qmp_capabilities" {
            caps_done = true;
        }
        // Both quit and its QMP alias system_powerdown must end the
        // connection loop. Without this the dispatch arm flips
        // quit_requested but reader.lines() keeps blocking, so a
        // peer that holds the socket open after system_powerdown
        // leaves the monitor thread parked indefinitely.
        if cmd == "quit" || cmd == "system_powerdown" {
            return true;
        }
    }
    false
}

/// Dispatch a command and return the JSON response.
pub fn dispatch(cmd: &str, svc: &Arc<Mutex<MonitorService>>) -> Value {
    let s = svc.lock().expect("monitor mutex poisoned");
    match cmd {
        "qmp_capabilities" => json!({"return": {}}),
        "query-version" => {
            // QMP `VersionInfo` shape: the `qemu` key carries the
            // version triple (major/minor/micro) and `package`
            // carries the project name. Keeping the key name
            // `qemu` rather than `machina` is what lets standard
            // QMP tooling (libvirt, qmp-shell, etc.) deserialise
            // this response — the greeting at the top of this
            // file mirrors the same triple under a `machina`
            // alias for the same reason.
            let (major, minor, micro) = parse_pkg_version();
            json!({
                "return": {
                    "qemu": {
                        "major": major,
                        "minor": minor,
                        "micro": micro,
                    },
                    "package": "machina",
                }
            })
        }
        "query-status" => {
            let running = s.query_status();
            json!({"return": {"running": running}})
        }
        "stop" => {
            drop(s);
            svc.lock().expect("monitor mutex poisoned").stop();
            json!({"return": {}})
        }
        "cont" => {
            s.cont();
            json!({"return": {}})
        }
        "quit" => {
            s.quit();
            json!({"return": {}})
        }
        "system_powerdown" => {
            // QMP-compatible alias for quit: request shutdown so the
            // emulator exits cleanly.
            s.quit();
            json!({"return": {}})
        }
        "query-cpus-fast" => {
            let cpus = s.query_cpus();
            let arr: Vec<Value> = cpus
                .iter()
                .map(|c| {
                    json!({
                        "cpu-index": c.cpu_index,
                        "qom-path": format!(
                            "/machine/cpu[{}]",
                            c.cpu_index
                        ),
                        "thread-id": 0,
                        "halted": c.halted,
                        "arch": c.arch,
                        "props": {
                            "core-id": c.cpu_index
                        },
                        "target": c.arch,
                    })
                })
                .collect();
            json!({"return": arr})
        }
        "system_reset" => {
            json!({
                "error": {
                    "class": "GenericError",
                    "desc": "system_reset not \
                             implemented (deferred)"
                }
            })
        }
        _ => {
            json!({
                "error": {
                    "class": "CommandNotFound",
                    "desc": format!(
                        "command '{}' not found",
                        cmd
                    )
                }
            })
        }
    }
}
