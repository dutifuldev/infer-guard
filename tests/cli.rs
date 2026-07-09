use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::tempdir;

fn meminfo(mem_available_kb: u64, swap_free_kb: u64) -> String {
    format!(
        "MemTotal: 999999999 kB\nMemAvailable: {mem_available_kb} kB\nSwapFree: {swap_free_kb} kB\n"
    )
}

fn wait_for_file(path: &Path) {
    for _ in 0..100 {
        if path.exists() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("timed out waiting for {}", path.display());
}

fn process_exists(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

fn wait_for_process_exit(pid: u32) {
    for _ in 0..100 {
        if !process_exists(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("process {pid} is still alive");
}

#[test]
fn doctor_prints_status() {
    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.arg("doctor");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("infer-guard doctor"));
}

#[test]
fn doctor_and_inspect_can_emit_json() {
    let mut doctor = Command::cargo_bin("infer-guard").unwrap();
    doctor.args(["doctor", "--json"]);
    doctor
        .assert()
        .success()
        .stdout(predicate::str::contains("\"earlyoom_active\""));

    let mut inspect = Command::cargo_bin("infer-guard").unwrap();
    inspect.args(["inspect", "--json"]);
    inspect
        .assert()
        .success()
        .stdout(predicate::str::contains("\"default_tools\""));
}

#[test]
fn inspect_prints_human_summary() {
    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.arg("inspect");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("infer-guard inspect"));
}

#[test]
fn run_returns_child_exit_code() {
    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "exit 7",
    ]);
    cmd.assert().code(7);
}

#[test]
fn run_returns_signal_exit_code() {
    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "kill -TERM $$",
    ]);
    cmd.assert().code(143);
}

#[test]
fn high_risk_profiles_require_earlyoom_unless_overridden() {
    let mut present = Command::cargo_bin("infer-guard").unwrap();
    present.env("INFER_GUARD_EARLYOOM_ACTIVE", "1");
    present.args([
        "run",
        "--profile",
        "vllm",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    present.assert().success();

    let mut blocked = Command::cargo_bin("infer-guard").unwrap();
    blocked.env("INFER_GUARD_EARLYOOM_ACTIVE", "0");
    blocked.args([
        "run",
        "--profile",
        "vllm",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    blocked
        .assert()
        .code(3)
        .stderr(predicate::str::contains("earlyoom is required"));

    let mut python_module = Command::cargo_bin("infer-guard").unwrap();
    python_module.env("INFER_GUARD_EARLYOOM_ACTIVE", "0");
    python_module.args([
        "run",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--",
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--help",
    ]);
    python_module
        .assert()
        .code(3)
        .stderr(predicate::str::contains("earlyoom is required"));

    let mut tgi = Command::cargo_bin("infer-guard").unwrap();
    tgi.env("INFER_GUARD_EARLYOOM_ACTIVE", "0");
    tgi.args([
        "run",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--",
        "text-generation-launcher",
        "--help",
    ]);
    tgi.assert()
        .code(3)
        .stderr(predicate::str::contains("earlyoom is required"));

    let mut allowed = Command::cargo_bin("infer-guard").unwrap();
    allowed.env("INFER_GUARD_EARLYOOM_ACTIVE", "0");
    allowed.args([
        "run",
        "--profile",
        "vllm",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    allowed.assert().success();
}

#[test]
fn run_rejects_unsupported_systemd_scope_flag() {
    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--allow-no-earlyoom",
        "--use-systemd-scope",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    cmd.assert()
        .code(2)
        .stderr(predicate::str::contains("systemd scope support"));

    let mut memory_high = Command::cargo_bin("infer-guard").unwrap();
    memory_high.args([
        "run",
        "--profile",
        "generic",
        "--allow-no-earlyoom",
        "--memory-high",
        "1G",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    memory_high
        .assert()
        .code(2)
        .stderr(predicate::str::contains("systemd scope support"));

    let mut memory_max = Command::cargo_bin("infer-guard").unwrap();
    memory_max.args([
        "run",
        "--profile",
        "generic",
        "--allow-no-earlyoom",
        "--memory-max",
        "1G",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    memory_max
        .assert()
        .code(2)
        .stderr(predicate::str::contains("systemd scope support"));
}

#[test]
fn run_allows_thresholds_equal_to_available_memory() {
    let dir = tempdir().unwrap();
    let meminfo_path = dir.path().join("meminfo");
    fs::write(&meminfo_path, meminfo(2048, 1024)).unwrap();

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.env("INFER_GUARD_MEMINFO_PATH", &meminfo_path);
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "2M",
        "--min-swap",
        "1M",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    cmd.assert().success();
}

#[test]
fn run_refuses_when_preflight_memory_is_below_floor() {
    let dir = tempdir().unwrap();
    let meminfo_path = dir.path().join("meminfo");
    fs::write(&meminfo_path, meminfo(1024, 0)).unwrap();

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.env("INFER_GUARD_MEMINFO_PATH", &meminfo_path);
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "2M",
        "--min-swap",
        "0",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    cmd.assert()
        .code(4)
        .stderr(predicate::str::contains("refusing to launch"));
}

#[test]
fn run_refuses_when_preflight_swap_is_below_floor() {
    let dir = tempdir().unwrap();
    let meminfo_path = dir.path().join("meminfo");
    fs::write(&meminfo_path, meminfo(10 * 1024, 0)).unwrap();

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.env("INFER_GUARD_MEMINFO_PATH", &meminfo_path);
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "1M",
        "--min-swap",
        "1M",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    cmd.assert()
        .code(4)
        .stderr(predicate::str::contains("SwapFree"));
}

#[test]
fn run_kills_process_group_when_memory_drops() {
    let dir = tempdir().unwrap();
    let meminfo_path = dir.path().join("meminfo");
    let event_log = dir.path().join("events.jsonl");
    fs::write(&meminfo_path, meminfo(10 * 1024 * 1024, 1024)).unwrap();

    let script = format!(
        "sleep 0.2; printf '{}' > {}; sleep 20",
        meminfo(1024, 1024).replace('\n', "\\n"),
        meminfo_path.display()
    );

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.env("INFER_GUARD_MEMINFO_PATH", &meminfo_path);
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "2M",
        "--min-swap",
        "0",
        "--poll",
        "50ms",
        "--term-grace",
        "100ms",
        "--allow-no-earlyoom",
        "--event-log",
        event_log.to_str().unwrap(),
        "--",
        "bash",
        "-lc",
        &script,
    ]);
    let started = Instant::now();
    cmd.assert().code(137);
    assert!(
        started.elapsed().as_secs() < 5,
        "guard should kill the sleeping process promptly"
    );
    let events = fs::read_to_string(event_log).unwrap();
    assert!(events.contains("memory_pressure_kill"));
}

#[test]
fn run_escalates_to_sigkill_when_child_ignores_sigterm() {
    let dir = tempdir().unwrap();
    let meminfo_path = dir.path().join("meminfo");
    fs::write(&meminfo_path, meminfo(10 * 1024 * 1024, 1024)).unwrap();
    let script = format!(
        "trap '' TERM; sleep 0.2; printf '{}' > {}; sleep 20",
        meminfo(1024, 1024).replace('\n', "\\n"),
        meminfo_path.display()
    );

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.env("INFER_GUARD_MEMINFO_PATH", &meminfo_path);
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "2M",
        "--min-swap",
        "0",
        "--poll",
        "50ms",
        "--term-grace",
        "800ms",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        &script,
    ]);
    let started = Instant::now();
    cmd.assert().code(137);
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_millis() >= 700,
        "guard should wait for TERM grace before KILL"
    );
    assert!(
        elapsed.as_secs() < 10,
        "guard should escalate instead of waiting for the child sleep"
    );
}

#[test]
fn run_cleans_process_group_when_guard_receives_term() {
    let dir = tempdir().unwrap();
    let child_pid_path = dir.path().join("child.pid");
    let script = format!(
        "echo $$ > {}; trap '' TERM; sleep 20",
        child_pid_path.display()
    );

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.args([
        "run",
        "--profile",
        "generic",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
        "--poll",
        "1s",
        "--term-grace",
        "100ms",
        "--allow-no-earlyoom",
        "--",
        "bash",
        "-lc",
        &script,
    ]);
    let mut guard = cmd.spawn().unwrap();
    wait_for_file(&child_pid_path);
    let child_pid: u32 = fs::read_to_string(&child_pid_path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    Command::new("kill")
        .arg("-TERM")
        .arg(guard.id().to_string())
        .assert()
        .success();
    let status = guard.wait().unwrap();

    assert_eq!(status.code(), Some(143));
    wait_for_process_exit(child_pid);
}

#[test]
fn run_rejects_invalid_durations_without_panic() {
    let mut poll = Command::cargo_bin("infer-guard").unwrap();
    poll.args([
        "run",
        "--profile",
        "generic",
        "--allow-no-earlyoom",
        "--poll",
        "-1",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    poll.assert()
        .code(2)
        .stderr(predicate::str::contains("invalid --poll"))
        .stderr(predicate::str::contains("panicked").not());

    let mut term_grace = Command::cargo_bin("infer-guard").unwrap();
    term_grace.args([
        "run",
        "--profile",
        "generic",
        "--allow-no-earlyoom",
        "--term-grace",
        "NaN",
        "--",
        "bash",
        "-lc",
        "exit 0",
    ]);
    term_grace
        .assert()
        .code(2)
        .stderr(predicate::str::contains("invalid --term-grace"))
        .stderr(predicate::str::contains("panicked").not());
}

#[test]
fn path_shim_resolves_real_binary_without_recursing() {
    let dir = tempdir().unwrap();
    let shim_dir = dir.path().join("shims");
    let real_dir = dir.path().join("real");
    fs::create_dir_all(&real_dir).unwrap();
    let real = real_dir.join("fake-vllm");
    fs::write(&real, "#!/usr/bin/env bash\necho real-fake-vllm \"$@\"\n").unwrap();
    fs::set_permissions(&real, fs::Permissions::from_mode(0o755)).unwrap();

    let mut install = Command::cargo_bin("infer-guard").unwrap();
    install.args([
        "install-shims",
        "--bin-dir",
        shim_dir.to_str().unwrap(),
        "--tool",
        "fake-vllm",
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
    ]);
    install.assert().success();

    let shim = shim_dir.join("fake-vllm");
    let mut run = Command::new(&shim);
    let system_path = std::env::var("PATH").unwrap_or_default();
    run.env(
        "PATH",
        format!(
            "{}:{}:{system_path}",
            shim_dir.display(),
            real_dir.display()
        ),
    );
    run.env("INFER_GUARD_ALLOW_NO_EARLYOOM", "1");
    run.arg("smoke");
    run.assert()
        .success()
        .stdout(predicate::str::contains("real-fake-vllm smoke"));
}

#[test]
fn path_shim_treats_generated_defaults_as_data() {
    let dir = tempdir().unwrap();
    let shim_dir = dir.path().join("shims");
    let real_dir = dir.path().join("real");
    let sentinel = dir.path().join("sentinel");
    let payload = format!("$(touch {})", sentinel.display());
    fs::create_dir_all(&real_dir).unwrap();
    let real = real_dir.join("fake-vllm");
    fs::write(&real, "#!/usr/bin/env bash\necho should-not-run\n").unwrap();
    fs::set_permissions(&real, fs::Permissions::from_mode(0o755)).unwrap();

    let mut install = Command::cargo_bin("infer-guard").unwrap();
    install
        .arg("install-shims")
        .arg("--bin-dir")
        .arg(&shim_dir)
        .args(["--tool", "fake-vllm", "--min-mem"])
        .arg(&payload)
        .args(["--min-swap", "0"]);
    install.assert().success();

    let shim = shim_dir.join("fake-vllm");
    let mut run = Command::new(&shim);
    let system_path = std::env::var("PATH").unwrap_or_default();
    run.env(
        "PATH",
        format!(
            "{}:{}:{system_path}",
            shim_dir.display(),
            real_dir.display()
        ),
    );
    run.env("INFER_GUARD_ALLOW_NO_EARLYOOM", "1");
    run.assert()
        .code(2)
        .stderr(predicate::str::contains("invalid --min-mem"));
    assert!(
        !sentinel.exists(),
        "generated shim default executed as shell code"
    );
}

#[test]
fn install_default_shims_and_refuse_unmanaged_collision() {
    let dir = tempdir().unwrap();
    let shim_dir = dir.path().join("shims");

    let mut install = Command::cargo_bin("infer-guard").unwrap();
    install.args([
        "install-shims",
        "--bin-dir",
        shim_dir.to_str().unwrap(),
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
    ]);
    install.assert().success();
    assert!(shim_dir.join("vllm").exists());
    assert!(shim_dir.join("llama-server").exists());

    fs::write(shim_dir.join("custom-tool"), "not managed\n").unwrap();
    let mut collision = Command::cargo_bin("infer-guard").unwrap();
    collision.args([
        "install-shims",
        "--bin-dir",
        shim_dir.to_str().unwrap(),
        "--tool",
        "custom-tool",
    ]);
    collision
        .assert()
        .code(2)
        .stderr(predicate::str::contains("refusing to replace"));

    let mut uninstall = Command::cargo_bin("infer-guard").unwrap();
    uninstall.args([
        "uninstall-shims",
        "--bin-dir",
        shim_dir.to_str().unwrap(),
        "--tool",
        "vllm",
    ]);
    uninstall.assert().success();
    assert!(!shim_dir.join("vllm").exists());
}

#[test]
fn wrap_and_unwrap_round_trip() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("vllm");
    fs::write(&target, "#!/usr/bin/env bash\necho wrapped-real \"$@\"\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

    let mut wrap = Command::cargo_bin("infer-guard").unwrap();
    wrap.args([
        "wrap",
        target.to_str().unwrap(),
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
    ]);
    wrap.assert().success();
    assert!(target.with_file_name("vllm.real").exists());

    let mut guarded = Command::new(&target);
    guarded.env("INFER_GUARD_ALLOW_NO_EARLYOOM", "1");
    guarded.arg("absolute");
    guarded
        .assert()
        .success()
        .stdout(predicate::str::contains("wrapped-real absolute"));

    let mut unwrap = Command::cargo_bin("infer-guard").unwrap();
    unwrap.args(["unwrap", target.to_str().unwrap()]);
    unwrap.assert().success();
    assert!(!target.with_file_name("vllm.real").exists());

    let mut restored = Command::new(&target);
    restored.arg("restored");
    restored
        .assert()
        .success()
        .stdout(predicate::str::contains("wrapped-real restored"));
}

#[test]
fn wrapped_high_risk_tool_still_requires_earlyoom() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("vllm");
    fs::write(&target, "#!/usr/bin/env bash\necho should-not-run\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

    let mut wrap = Command::cargo_bin("infer-guard").unwrap();
    wrap.args([
        "wrap",
        target.to_str().unwrap(),
        "--min-mem",
        "1M",
        "--min-swap",
        "0",
    ]);
    wrap.assert().success();

    let mut guarded = Command::new(&target);
    guarded.env("INFER_GUARD_EARLYOOM_ACTIVE", "0");
    guarded
        .assert()
        .code(3)
        .stderr(predicate::str::contains("earlyoom is required"));
}

#[test]
fn absolute_wrapper_treats_generated_defaults_as_data() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("vllm");
    let sentinel = dir.path().join("sentinel");
    let payload = format!("$(touch {})", sentinel.display());
    fs::write(&target, "#!/usr/bin/env bash\necho should-not-run\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

    let mut wrap = Command::cargo_bin("infer-guard").unwrap();
    wrap.arg("wrap")
        .arg(&target)
        .args(["--min-mem"])
        .arg(&payload)
        .args(["--min-swap", "0"]);
    wrap.assert().success();

    let mut guarded = Command::new(&target);
    guarded.env("INFER_GUARD_ALLOW_NO_EARLYOOM", "1");
    guarded
        .assert()
        .code(2)
        .stderr(predicate::str::contains("invalid --min-mem"));
    assert!(
        !sentinel.exists(),
        "generated wrapper default executed as shell code"
    );
}

#[test]
fn wrap_and_unwrap_validate_bad_inputs() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing");
    let mut wrap_missing = Command::cargo_bin("infer-guard").unwrap();
    wrap_missing.args(["wrap", missing.to_str().unwrap()]);
    wrap_missing
        .assert()
        .code(2)
        .stderr(predicate::str::contains("cannot wrap missing path"));

    let directory = dir.path().join("directory");
    fs::create_dir(&directory).unwrap();
    let mut wrap_directory = Command::cargo_bin("infer-guard").unwrap();
    wrap_directory.args(["wrap", directory.to_str().unwrap()]);
    wrap_directory
        .assert()
        .code(2)
        .stderr(predicate::str::contains("cannot wrap non-file path"));

    let target = dir.path().join("not-executable");
    fs::write(&target, "plain text\n").unwrap();
    let mut wrap_non_executable = Command::cargo_bin("infer-guard").unwrap();
    wrap_non_executable.args(["wrap", target.to_str().unwrap()]);
    wrap_non_executable
        .assert()
        .code(2)
        .stderr(predicate::str::contains("cannot wrap non-executable path"));

    let mut unwrap_plain = Command::cargo_bin("infer-guard").unwrap();
    unwrap_plain.args(["unwrap", target.to_str().unwrap()]);
    unwrap_plain
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "not an infer-guard managed wrapper",
        ));
}

#[test]
fn wrap_refuses_existing_real_path_without_force() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("vllm");
    fs::write(&target, "#!/usr/bin/env bash\necho target\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(dir.path().join("vllm.real"), "already exists\n").unwrap();

    let mut cmd = Command::cargo_bin("infer-guard").unwrap();
    cmd.args(["wrap", target.to_str().unwrap()]);
    cmd.assert()
        .code(2)
        .stderr(predicate::str::contains("refusing to overwrite"));
}
