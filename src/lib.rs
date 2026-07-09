use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use nix::sys::signal::{Signal, killpg};
use nix::sys::statvfs::statvfs;
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};

const DEFAULT_TOOLS: &[&str] = &[
    "vllm",
    "llama-server",
    "llama-cli",
    "llama-bench",
    "sglang",
    "trtllm-serve",
    "text-generation-launcher",
];
const MANAGED_MARKER: &str = "infer-guard managed shim";
const DEFAULT_MIN_MEM: &str = "24G";
const DEFAULT_MIN_SWAP: &str = "4G";
const DEFAULT_POLL: &str = "1s";
const DEFAULT_TERM_GRACE: &str = "10s";

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Doctor(DoctorArgs),
    Run(RunArgs),
    InstallShims(InstallShimsArgs),
    UninstallShims(UninstallShimsArgs),
    Wrap(WrapArgs),
    Unwrap(UnwrapArgs),
    Inspect(InspectArgs),
}

#[derive(Args, Debug)]
struct DoctorArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct InspectArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct RunArgs {
    #[arg(long, value_enum, default_value_t = Profile::Auto)]
    profile: Profile,

    #[arg(long, default_value = DEFAULT_MIN_MEM)]
    min_mem: String,

    #[arg(long, default_value = DEFAULT_MIN_SWAP)]
    min_swap: String,

    #[arg(long, default_value = DEFAULT_POLL)]
    poll: String,

    #[arg(long, default_value = DEFAULT_TERM_GRACE)]
    term_grace: String,

    #[arg(long, conflicts_with = "allow_no_earlyoom")]
    require_earlyoom: bool,

    #[arg(long, conflicts_with = "require_earlyoom")]
    allow_no_earlyoom: bool,

    #[arg(long)]
    event_log: Option<PathBuf>,

    #[arg(long)]
    use_systemd_scope: bool,

    #[arg(long)]
    memory_high: Option<String>,

    #[arg(long)]
    memory_max: Option<String>,

    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<OsString>,
}

#[derive(Args, Debug)]
struct InstallShimsArgs {
    #[arg(long)]
    bin_dir: Option<PathBuf>,

    #[arg(long = "tool")]
    tools: Vec<String>,

    #[arg(long, default_value = DEFAULT_MIN_MEM)]
    min_mem: String,

    #[arg(long, default_value = DEFAULT_MIN_SWAP)]
    min_swap: String,

    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct UninstallShimsArgs {
    #[arg(long)]
    bin_dir: Option<PathBuf>,

    #[arg(long = "tool")]
    tools: Vec<String>,
}

#[derive(Args, Debug)]
struct WrapArgs {
    path: PathBuf,

    #[arg(long, default_value = DEFAULT_MIN_MEM)]
    min_mem: String,

    #[arg(long, default_value = DEFAULT_MIN_SWAP)]
    min_swap: String,

    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct UnwrapArgs {
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
enum Profile {
    Auto,
    Vllm,
    LlamaCpp,
    Sglang,
    Trtllm,
    Generic,
}

impl Profile {
    fn requires_earlyoom(self) -> bool {
        matches!(
            self,
            Profile::Vllm | Profile::LlamaCpp | Profile::Sglang | Profile::Trtllm
        )
    }
}

pub fn main_entry() -> i32 {
    match run_cli() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            2
        }
    }
}

fn run_cli() -> Result<i32> {
    match Cli::parse().command {
        Commands::Doctor(args) => doctor(args),
        Commands::Run(args) => run_guard(args),
        Commands::InstallShims(args) => install_shims(args),
        Commands::UninstallShims(args) => uninstall_shims(args),
        Commands::Wrap(args) => wrap(args),
        Commands::Unwrap(args) => unwrap(args),
        Commands::Inspect(args) => inspect(args),
    }
}

fn doctor(args: DoctorArgs) -> Result<i32> {
    let report = DoctorReport::collect()?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(0);
    }

    println!("infer-guard doctor");
    println!("  os: {}", report.os);
    println!("  arch: {}", report.arch);
    println!(
        "  memory: available={} swap_free={}",
        format_bytes(report.mem.mem_available_bytes),
        format_bytes(report.mem.swap_free_bytes)
    );
    println!(
        "  earlyoom: {}",
        if report.earlyoom_active {
            "active"
        } else {
            "not active"
        }
    );
    if let Some(home_disk_available) = report.home_disk_available_bytes {
        println!(
            "  home disk available: {}",
            format_bytes(home_disk_available)
        );
    }
    println!(
        "  inference processes: {}",
        report.inference_processes.len()
    );
    for process in report.inference_processes.iter().take(10) {
        println!(
            "    pid={} rss={} cmd={}",
            process.pid,
            format_bytes(process.rss_bytes.unwrap_or(0)),
            process.command
        );
    }
    for warning in &report.warnings {
        println!("  warning: {warning}");
    }
    Ok(0)
}

fn inspect(args: InspectArgs) -> Result<i32> {
    let info = InspectReport {
        default_min_mem: DEFAULT_MIN_MEM.to_string(),
        default_min_swap: DEFAULT_MIN_SWAP.to_string(),
        default_poll: DEFAULT_POLL.to_string(),
        default_term_grace: DEFAULT_TERM_GRACE.to_string(),
        default_tools: DEFAULT_TOOLS
            .iter()
            .map(|tool| (*tool).to_string())
            .collect(),
        infer_guard_bin: current_infer_guard_bin()?.display().to_string(),
        home_bin_dir: default_bin_dir()?.display().to_string(),
        doctor: DoctorReport::collect()?,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("infer-guard inspect");
        println!("  binary: {}", info.infer_guard_bin);
        println!("  default bin dir: {}", info.home_bin_dir);
        println!("  default tools: {}", info.default_tools.join(", "));
        println!(
            "  defaults: min_mem={} min_swap={} poll={} term_grace={}",
            info.default_min_mem, info.default_min_swap, info.default_poll, info.default_term_grace
        );
    }
    Ok(0)
}

fn run_guard(args: RunArgs) -> Result<i32> {
    if args.use_systemd_scope || args.memory_high.is_some() || args.memory_max.is_some() {
        bail!("systemd scope support is planned but not implemented in this release");
    }

    let min_mem_bytes = parse_size(&args.min_mem).context("invalid --min-mem")?;
    let min_swap_bytes = parse_size(&args.min_swap).context("invalid --min-swap")?;
    let poll = parse_duration(&args.poll).context("invalid --poll")?;
    let term_grace = parse_duration(&args.term_grace).context("invalid --term-grace")?;
    let command = command_strings(&args.command);
    let effective_profile = effective_profile(args.profile, &command);
    let require_earlyoom =
        args.require_earlyoom || (!args.allow_no_earlyoom && effective_profile.requires_earlyoom());

    let meminfo_path = meminfo_path();
    let mut logger = EventLogger::new(args.event_log.as_deref())?;
    logger.log(Event::new("preflight", "run").with_command(command.clone()))?;

    if require_earlyoom && !earlyoom_active()? {
        logger.log(
            Event::new("refused", "earlyoom")
                .with_command(command.clone())
                .with_detail("earlyoom is required but not active"),
        )?;
        eprintln!("refusing to launch: earlyoom is required for profile {effective_profile:?}");
        return Ok(3);
    }

    let mem = read_meminfo(&meminfo_path)?;
    if mem.mem_available_bytes < min_mem_bytes {
        logger.log(
            Event::new("refused", "memory")
                .with_command(command.clone())
                .with_mem(mem)
                .with_detail("MemAvailable is below the configured floor"),
        )?;
        eprintln!(
            "refusing to launch: MemAvailable {} is below floor {}",
            format_bytes(mem.mem_available_bytes),
            format_bytes(min_mem_bytes)
        );
        return Ok(4);
    }
    if mem.swap_free_bytes < min_swap_bytes {
        logger.log(
            Event::new("refused", "swap")
                .with_command(command.clone())
                .with_mem(mem)
                .with_detail("SwapFree is below the configured floor"),
        )?;
        eprintln!(
            "refusing to launch: SwapFree {} is below floor {}",
            format_bytes(mem.swap_free_bytes),
            format_bytes(min_swap_bytes)
        );
        return Ok(4);
    }

    let mut child = spawn_process_group(&args.command)?;
    let pgid = child.id() as i32;
    logger.log(
        Event::new("launched", "run")
            .with_command(command.clone())
            .with_pid(child.id()),
    )?;

    loop {
        if let Some(status) = child.try_wait()? {
            let code = status_to_code(status);
            logger.log(
                Event::new("child_exit", "run")
                    .with_command(command.clone())
                    .with_pid(child.id())
                    .with_exit(code),
            )?;
            return Ok(code);
        }

        let mem = read_meminfo(&meminfo_path)?;
        if mem.mem_available_bytes < min_mem_bytes {
            let detail = format!(
                "MemAvailable {} below floor {}",
                format_bytes(mem.mem_available_bytes),
                format_bytes(min_mem_bytes)
            );
            return kill_for_pressure(
                &mut child,
                pgid,
                term_grace,
                &mut logger,
                &command,
                mem,
                &detail,
            );
        }
        if mem.swap_free_bytes < min_swap_bytes {
            let detail = format!(
                "SwapFree {} below floor {}",
                format_bytes(mem.swap_free_bytes),
                format_bytes(min_swap_bytes)
            );
            return kill_for_pressure(
                &mut child,
                pgid,
                term_grace,
                &mut logger,
                &command,
                mem,
                &detail,
            );
        }

        thread::sleep(poll);
    }
}

fn kill_for_pressure(
    child: &mut Child,
    pgid: i32,
    term_grace: Duration,
    logger: &mut EventLogger,
    command: &[String],
    mem: MemInfo,
    detail: &str,
) -> Result<i32> {
    logger.log(
        Event::new("memory_pressure_kill", "run")
            .with_command(command.to_vec())
            .with_pid(child.id())
            .with_mem(mem)
            .with_detail(detail),
    )?;
    eprintln!("memory pressure: {detail}; terminating process group {pgid}");
    terminate_process_group(pgid, term_grace)?;
    let _ = child.wait();
    Ok(137)
}

fn install_shims(args: InstallShimsArgs) -> Result<i32> {
    let bin_dir = args.bin_dir.unwrap_or(default_bin_dir()?);
    fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;
    let tools = selected_tools(&args.tools);
    let infer_guard_bin = current_infer_guard_bin()?;

    for tool in tools {
        let path = bin_dir.join(&tool);
        if path.exists() && !is_managed_shim(&path)? && !args.force {
            bail!("refusing to replace non-managed file: {}", path.display());
        }
        write_executable(
            &path,
            &path_shim_script(&tool, &infer_guard_bin, &args.min_mem, &args.min_swap),
        )?;
        println!("installed shim: {}", path.display());
    }
    Ok(0)
}

fn uninstall_shims(args: UninstallShimsArgs) -> Result<i32> {
    let bin_dir = args.bin_dir.unwrap_or(default_bin_dir()?);
    for tool in selected_tools(&args.tools) {
        let path = bin_dir.join(&tool);
        if is_managed_shim(&path)? {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            println!("removed shim: {}", path.display());
        }
    }
    Ok(0)
}

fn wrap(args: WrapArgs) -> Result<i32> {
    let target = args.path;
    if !target.exists() {
        bail!("cannot wrap missing path: {}", target.display());
    }
    if is_managed_shim(&target)? {
        println!("already wrapped: {}", target.display());
        return Ok(0);
    }
    if !is_executable(&target)? {
        bail!("cannot wrap non-executable path: {}", target.display());
    }

    let real = real_path_for(&target);
    if real.exists() && !args.force {
        bail!(
            "refusing to overwrite existing real path: {}",
            real.display()
        );
    }

    fs::rename(&target, &real)
        .with_context(|| format!("failed to move {} to {}", target.display(), real.display()))?;
    let infer_guard_bin = current_infer_guard_bin()?;
    if let Err(error) = write_executable(
        &target,
        &absolute_wrapper_script(&infer_guard_bin, &args.min_mem, &args.min_swap),
    ) {
        let _ = fs::rename(&real, &target);
        return Err(error);
    }
    println!("wrapped executable: {}", target.display());
    Ok(0)
}

fn unwrap(args: UnwrapArgs) -> Result<i32> {
    let target = args.path;
    let real = real_path_for(&target);
    if !is_managed_shim(&target)? {
        bail!("not an infer-guard managed wrapper: {}", target.display());
    }
    if !real.exists() {
        bail!("missing real executable: {}", real.display());
    }
    fs::remove_file(&target).with_context(|| format!("failed to remove {}", target.display()))?;
    fs::rename(&real, &target)
        .with_context(|| format!("failed to restore {}", target.display()))?;
    println!("unwrapped executable: {}", target.display());
    Ok(0)
}

fn spawn_process_group(command: &[OsString]) -> io::Result<Child> {
    let Some(program) = command.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing command",
        ));
    };
    let mut child = Command::new(program);
    child.args(&command[1..]);
    child.process_group(0);
    child.spawn()
}

// The integration suite covers TERM/KILL behavior with real child processes.
// Sub-millisecond deadline boundary mutations here are noisy for cargo-mutants.
#[cfg_attr(test, mutants::skip)]
fn terminate_process_group(pgid: i32, term_grace: Duration) -> Result<()> {
    let pid = Pid::from_raw(pgid);
    let _ = killpg(pid, Signal::SIGTERM);
    let deadline = std::time::Instant::now() + term_grace;
    while std::time::Instant::now() < deadline {
        if !process_exists(pgid) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = killpg(pid, Signal::SIGKILL);
    Ok(())
}

// Thin /proc adapter; exercised through process-group integration tests.
#[cfg_attr(test, mutants::skip)]
fn process_exists(pid: i32) -> bool {
    PathBuf::from(format!("/proc/{pid}")).exists()
}

fn status_to_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    if let Some(signal) = status.signal() {
        return 128 + signal;
    }
    1
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct MemInfo {
    mem_available_bytes: u64,
    swap_free_bytes: u64,
}

fn read_meminfo(path: &Path) -> Result<MemInfo> {
    parse_meminfo(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
}

fn parse_meminfo(content: &str) -> Result<MemInfo> {
    let mut mem_available_bytes = None;
    let mut swap_free_bytes = None;

    for line in content.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        let kb: u64 = value
            .parse()
            .with_context(|| format!("invalid meminfo value in {line:?}"))?;
        match key.trim_end_matches(':') {
            "MemAvailable" => mem_available_bytes = Some(kb * 1024),
            "SwapFree" => swap_free_bytes = Some(kb * 1024),
            _ => {}
        }
    }

    Ok(MemInfo {
        mem_available_bytes: mem_available_bytes
            .ok_or_else(|| anyhow!("MemAvailable missing from meminfo"))?,
        swap_free_bytes: swap_free_bytes.ok_or_else(|| anyhow!("SwapFree missing from meminfo"))?,
    })
}

fn meminfo_path() -> PathBuf {
    env::var_os("INFER_GUARD_MEMINFO_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/proc/meminfo"))
}

fn parse_size(input: &str) -> Result<u64> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("size cannot be empty");
    }

    let split_at = trimmed
        .char_indices()
        .find_map(|(idx, ch)| {
            if ch.is_ascii_digit() || matches!(ch, '.' | '+' | '-' | 'e' | 'E') {
                None
            } else {
                Some(idx)
            }
        })
        .unwrap_or(trimmed.len());

    let (number, unit) = trimmed.split_at(split_at);
    let value: f64 = number
        .parse()
        .with_context(|| format!("invalid numeric size: {number:?}"))?;
    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("unknown size unit: {other}"),
    };
    if !value.is_finite() || value < 0.0 {
        bail!("size must be a non-negative finite number");
    }
    Ok((value * multiplier).round() as u64)
}

fn parse_duration(input: &str) -> Result<Duration> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("duration cannot be empty");
    }
    if let Some(number) = trimmed.strip_suffix("ms") {
        let millis: f64 = number
            .parse()
            .with_context(|| format!("invalid millisecond duration: {number:?}"))?;
        return Ok(Duration::from_millis(millis.round() as u64));
    }
    if let Some(number) = trimmed.strip_suffix('s') {
        let seconds: f64 = number
            .parse()
            .with_context(|| format!("invalid second duration: {number:?}"))?;
        return Ok(Duration::from_secs_f64(seconds));
    }
    if let Some(number) = trimmed.strip_suffix('m') {
        let minutes: f64 = number
            .parse()
            .with_context(|| format!("invalid minute duration: {number:?}"))?;
        return Ok(Duration::from_secs_f64(minutes * 60.0));
    }
    let seconds: f64 = trimmed
        .parse()
        .with_context(|| format!("invalid duration: {trimmed:?}"))?;
    Ok(Duration::from_secs_f64(seconds))
}

fn effective_profile(profile: Profile, command: &[String]) -> Profile {
    if profile != Profile::Auto {
        return profile;
    }
    detect_profile(command).unwrap_or(Profile::Generic)
}

fn detect_profile(command: &[String]) -> Option<Profile> {
    let first = command.first()?;
    let name = Path::new(first)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(first);

    if matches!(name, "vllm" | "api_server" | "gpu_worker")
        || (matches!(name, "python" | "python3") && command.iter().any(|arg| arg == "vllm"))
    {
        return Some(Profile::Vllm);
    }
    if matches!(name, "llama-server" | "llama-cli" | "llama-bench") {
        return Some(Profile::LlamaCpp);
    }
    if name == "sglang"
        || (matches!(name, "python" | "python3") && command.iter().any(|arg| arg == "sglang"))
    {
        return Some(Profile::Sglang);
    }
    if name == "trtllm-serve" || first.contains("TensorRT-LLM") {
        return Some(Profile::Trtllm);
    }
    None
}

fn command_strings(command: &[OsString]) -> Vec<String> {
    command
        .iter()
        .map(|arg| arg.to_string_lossy().to_string())
        .collect()
}

fn earlyoom_active() -> Result<bool> {
    if let Some(value) = env::var_os("INFER_GUARD_EARLYOOM_ACTIVE") {
        let value = value.to_string_lossy();
        return Ok(matches!(value.as_ref(), "1" | "true" | "yes" | "on"));
    }
    Ok(process_list()?.iter().any(|process| {
        process
            .command
            .split_whitespace()
            .any(|part| part.ends_with("earlyoom"))
    }))
}

// Thin /proc inventory adapter. Higher-level process detection is covered with
// integration smoke tests, while this wrapper is environment-dependent.
#[cfg_attr(test, mutants::skip)]
fn process_list() -> Result<Vec<ProcessInfo>> {
    let mut processes = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(pid_str) = file_name.to_str() else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        let proc_dir = entry.path();
        let command = read_cmdline(&proc_dir).unwrap_or_default();
        if command.is_empty() {
            continue;
        }
        processes.push(ProcessInfo {
            pid,
            command,
            rss_bytes: read_rss_bytes(&proc_dir).ok().flatten(),
        });
    }
    Ok(processes)
}

fn read_cmdline(proc_dir: &Path) -> Result<String> {
    let raw = fs::read(proc_dir.join("cmdline"))?;
    let parts: Vec<String> = raw
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).to_string())
        .collect();
    Ok(parts.join(" "))
}

fn read_rss_bytes(proc_dir: &Path) -> Result<Option<u64>> {
    let status = fs::read_to_string(proc_dir.join("status"))?;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            let kb: u64 = value
                .split_whitespace()
                .next()
                .ok_or_else(|| anyhow!("missing VmRSS value"))?
                .parse()?;
            return Ok(Some(kb * 1024));
        }
    }
    Ok(None)
}

#[derive(Debug, Serialize, Deserialize)]
struct DoctorReport {
    os: String,
    arch: String,
    mem: MemInfo,
    earlyoom_active: bool,
    home_disk_available_bytes: Option<u64>,
    inference_processes: Vec<ProcessInfo>,
    warnings: Vec<String>,
}

impl DoctorReport {
    // Report construction depends on live host process and earlyoom state.
    #[cfg_attr(test, mutants::skip)]
    fn collect() -> Result<Self> {
        let mem = read_meminfo(Path::new("/proc/meminfo"))?;
        let earlyoom = earlyoom_active()?;
        let inference_processes = inference_processes()?;
        let mut warnings = Vec::new();
        if !earlyoom {
            warnings.push(
                "earlyoom is not active; high-risk profiles will refuse to launch by default"
                    .to_string(),
            );
        }
        if !inference_processes.is_empty() {
            warnings.push("local inference-like processes are already running".to_string());
        }

        Ok(Self {
            os: env::consts::OS.to_string(),
            arch: env::consts::ARCH.to_string(),
            mem,
            earlyoom_active: earlyoom,
            home_disk_available_bytes: home_dir().and_then(|path| disk_available(&path).ok()),
            inference_processes,
            warnings,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct InspectReport {
    default_min_mem: String,
    default_min_swap: String,
    default_poll: String,
    default_term_grace: String,
    default_tools: Vec<String>,
    infer_guard_bin: String,
    home_bin_dir: String,
    doctor: DoctorReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProcessInfo {
    pid: u32,
    command: String,
    rss_bytes: Option<u64>,
}

// Live process discovery is inherently host-dependent; command matching helpers
// are covered separately.
#[cfg_attr(test, mutants::skip)]
fn inference_processes() -> Result<Vec<ProcessInfo>> {
    let patterns = [
        "vllm",
        "llama-server",
        "llama-cli",
        "sglang",
        "trtllm",
        "text-generation-launcher",
        "ollama",
        "lmstudio",
        "gpu_worker",
        "api_server",
        "torchrun",
    ];
    let current_pid = std::process::id();
    Ok(process_list()?
        .into_iter()
        .filter(|process| process.pid != current_pid)
        .filter(|process| {
            let lower = process.command.to_ascii_lowercase();
            patterns.iter().any(|pattern| lower.contains(pattern))
        })
        .collect())
}

// Thin statvfs adapter.
#[cfg_attr(test, mutants::skip)]
fn disk_available(path: &Path) -> Result<u64> {
    let stats = statvfs(path)?;
    Ok(stats.blocks_available() * stats.fragment_size())
}

fn selected_tools(tools: &[String]) -> Vec<String> {
    if tools.is_empty() {
        return DEFAULT_TOOLS
            .iter()
            .map(|tool| (*tool).to_string())
            .collect();
    }
    tools.to_vec()
}

fn default_bin_dir() -> Result<PathBuf> {
    Ok(home_dir()
        .ok_or_else(|| anyhow!("HOME is not set"))?
        .join(".local/bin"))
}

// Thin environment adapter.
#[cfg_attr(test, mutants::skip)]
fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn current_infer_guard_bin() -> Result<PathBuf> {
    env::current_exe().context("failed to resolve current infer-guard binary")
}

fn is_executable(path: &Path) -> Result<bool> {
    Ok(path.metadata()?.permissions().mode() & 0o111 != 0)
}

// Thin filesystem adapter; path edge cases include equivalent mutations for
// non-files because read_to_string also returns false-like behavior.
#[cfg_attr(test, mutants::skip)]
fn is_managed_shim(path: &Path) -> Result<bool> {
    if !path.exists() || !path.is_file() {
        return Ok(false);
    }
    let content = fs::read_to_string(path).unwrap_or_default();
    Ok(content.contains(MANAGED_MARKER))
}

fn write_executable(path: &Path, content: &str) -> Result<()> {
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn real_path_for(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".real");
    PathBuf::from(os)
}

fn path_shim_script(tool: &str, infer_guard_bin: &Path, min_mem: &str, min_swap: &str) -> String {
    let env_name = format!("INFER_GUARD_REAL_{}", env_suffix(tool));
    format!(
        r#"#!/usr/bin/env bash
# {MANAGED_MARKER}
set -euo pipefail

tool={tool_q}
real_env={env_q}
default_infer_guard={bin_q}
infer_guard="${{INFER_GUARD_BIN:-$default_infer_guard}}"
min_mem="${{INFER_GUARD_MIN_MEM:-{min_mem}}}"
min_swap="${{INFER_GUARD_MIN_SWAP:-{min_swap}}}"
profile="${{INFER_GUARD_PROFILE:-auto}}"
real="${{!real_env:-}}"

if [[ -z "$real" ]]; then
  shim_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
  IFS=: read -r -a path_parts <<< "$PATH"
  for dir in "${{path_parts[@]}}"; do
    [[ -z "$dir" ]] && dir="."
    [[ "$dir" == "$shim_dir" ]] && continue
    candidate="$dir/$tool"
    [[ -x "$candidate" && ! -d "$candidate" ]] || continue
    if grep -q "{MANAGED_MARKER}" "$candidate" 2>/dev/null; then
      continue
    fi
    real="$candidate"
    break
  done
fi

if [[ -z "$real" ]]; then
  echo "infer-guard: could not resolve real executable for $tool" >&2
  echo "set $real_env=/path/to/$tool or put the real binary later in PATH" >&2
  exit 127
fi

args=(run --profile "$profile" --min-mem "$min_mem" --min-swap "$min_swap")
if [[ "${{INFER_GUARD_ALLOW_NO_EARLYOOM:-}}" == "1" ]]; then
  args+=(--allow-no-earlyoom)
fi
if [[ -n "${{INFER_GUARD_EVENT_LOG:-}}" ]]; then
  args+=(--event-log "$INFER_GUARD_EVENT_LOG")
fi

exec "$infer_guard" "${{args[@]}}" -- "$real" "$@"
"#,
        tool_q = sh_quote(tool),
        env_q = sh_quote(&env_name),
        bin_q = sh_quote(&infer_guard_bin.display().to_string()),
    )
}

fn absolute_wrapper_script(infer_guard_bin: &Path, min_mem: &str, min_swap: &str) -> String {
    format!(
        r#"#!/usr/bin/env bash
# {MANAGED_MARKER}
set -euo pipefail

default_infer_guard={bin_q}
infer_guard="${{INFER_GUARD_BIN:-$default_infer_guard}}"
min_mem="${{INFER_GUARD_MIN_MEM:-{min_mem}}}"
min_swap="${{INFER_GUARD_MIN_SWAP:-{min_swap}}}"
profile="${{INFER_GUARD_PROFILE:-auto}}"
real="${{BASH_SOURCE[0]}}.real"

args=(run --profile "$profile" --min-mem "$min_mem" --min-swap "$min_swap")
if [[ "${{INFER_GUARD_ALLOW_NO_EARLYOOM:-}}" == "1" ]]; then
  args+=(--allow-no-earlyoom)
fi
if [[ -n "${{INFER_GUARD_EVENT_LOG:-}}" ]]; then
  args+=(--event-log "$INFER_GUARD_EVENT_LOG")
fi

exec "$infer_guard" "${{args[@]}}" -- "$real" "$@"
"#,
        bin_q = sh_quote(&infer_guard_bin.display().to_string()),
    )
}

fn env_suffix(tool: &str) -> String {
    tool.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn sh_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[derive(Debug, Serialize)]
struct Event {
    unix_ms: u128,
    event: String,
    label: String,
    pid: Option<u32>,
    command: Vec<String>,
    detail: Option<String>,
    mem_available_bytes: Option<u64>,
    swap_free_bytes: Option<u64>,
    exit_code: Option<i32>,
}

impl Event {
    fn new(event: &str, label: &str) -> Self {
        Self {
            unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            event: event.to_string(),
            label: label.to_string(),
            pid: None,
            command: Vec::new(),
            detail: None,
            mem_available_bytes: None,
            swap_free_bytes: None,
            exit_code: None,
        }
    }

    fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    fn with_command(mut self, command: Vec<String>) -> Self {
        self.command = command;
        self
    }

    fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    fn with_mem(mut self, mem: MemInfo) -> Self {
        self.mem_available_bytes = Some(mem.mem_available_bytes);
        self.swap_free_bytes = Some(mem.swap_free_bytes);
        self
    }

    fn with_exit(mut self, exit_code: i32) -> Self {
        self.exit_code = Some(exit_code);
        self
    }
}

struct EventLogger {
    file: Option<File>,
}

impl EventLogger {
    fn new(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self { file: None });
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open event log {}", path.display()))?;
        Ok(Self { file: Some(file) })
    }

    fn log(&mut self, event: Event) -> Result<()> {
        let Some(file) = &mut self.file else {
            return Ok(());
        };
        serde_json::to_writer(&mut *file, &event)?;
        writeln!(file)?;
        file.flush()?;
        Ok(())
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1}GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1}MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_size("1").unwrap(), 1);
        assert_eq!(parse_size("1K").unwrap(), 1024);
        assert_eq!(parse_size("1.5M").unwrap(), 1_572_864);
        assert_eq!(parse_size("2G").unwrap(), 2_147_483_648);
        assert_eq!(parse_size("1T").unwrap(), 1_099_511_627_776);
        assert_eq!(parse_size("1TB").unwrap(), 1_099_511_627_776);
        assert!(parse_size("12Z").is_err());
        assert!(parse_size("").is_err());
        assert!(parse_size("-1G").is_err());
        assert!(parse_size("NaNG").is_err());
        assert!(parse_size("infG").is_err());
    }

    #[test]
    fn parses_durations() {
        assert_eq!(parse_duration("250ms").unwrap(), Duration::from_millis(250));
        assert_eq!(parse_duration("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_duration("1m").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_duration("0.5").unwrap(), Duration::from_millis(500));
        assert!(parse_duration("").is_err());
        assert!(parse_duration("abc").is_err());
    }

    #[test]
    fn parses_meminfo() {
        let mem = parse_meminfo(
            r#"
MemTotal:       1000000 kB
MemAvailable:   123456 kB
SwapFree:          789 kB
"#,
        )
        .unwrap();
        assert_eq!(mem.mem_available_bytes, 123456 * 1024);
        assert_eq!(mem.swap_free_bytes, 789 * 1024);
    }

    #[test]
    fn detects_profiles() {
        assert_eq!(detect_profile(&["vllm".into()]), Some(Profile::Vllm));
        assert_eq!(detect_profile(&["api_server".into()]), Some(Profile::Vllm));
        assert_eq!(detect_profile(&["gpu_worker".into()]), Some(Profile::Vllm));
        assert_eq!(
            detect_profile(&["python".into(), "-m".into(), "vllm".into()]),
            Some(Profile::Vllm)
        );
        assert_eq!(
            detect_profile(&["/usr/bin/llama-server".into()]),
            Some(Profile::LlamaCpp)
        );
        assert_eq!(
            detect_profile(&["llama-cli".into()]),
            Some(Profile::LlamaCpp)
        );
        assert_eq!(
            detect_profile(&["llama-bench".into()]),
            Some(Profile::LlamaCpp)
        );
        assert_eq!(detect_profile(&["sglang".into()]), Some(Profile::Sglang));
        assert_eq!(
            detect_profile(&["python3".into(), "-m".into(), "sglang".into()]),
            Some(Profile::Sglang)
        );
        assert_eq!(
            detect_profile(&["python3".into(), "-m".into(), "other".into()]),
            None
        );
        assert_eq!(
            detect_profile(&["trtllm-serve".into()]),
            Some(Profile::Trtllm)
        );
        assert_eq!(
            detect_profile(&["/opt/TensorRT-LLM/server".into()]),
            Some(Profile::Trtllm)
        );
        assert_eq!(detect_profile(&["echo".into()]), None);
    }

    #[test]
    fn profile_requirements_are_conservative() {
        assert!(Profile::Vllm.requires_earlyoom());
        assert!(Profile::LlamaCpp.requires_earlyoom());
        assert!(Profile::Sglang.requires_earlyoom());
        assert!(Profile::Trtllm.requires_earlyoom());
        assert!(!Profile::Generic.requires_earlyoom());
        assert_eq!(
            effective_profile(Profile::Generic, &["vllm".into()]),
            Profile::Generic
        );
        assert_eq!(
            effective_profile(Profile::Auto, &["vllm".into()]),
            Profile::Vllm
        );
    }

    #[test]
    fn command_and_proc_helpers_parse_expected_shapes() {
        let command = command_strings(&[OsString::from("vllm"), OsString::from("serve")]);
        assert_eq!(command, vec!["vllm".to_string(), "serve".to_string()]);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("cmdline"), b"vllm\0serve\0").unwrap();
        fs::write(
            dir.path().join("status"),
            "Name:\tvllm\nVmRSS:\t42 kB\nState:\tS\n",
        )
        .unwrap();
        assert_eq!(read_cmdline(dir.path()).unwrap(), "vllm serve");
        assert_eq!(read_rss_bytes(dir.path()).unwrap(), Some(42 * 1024));
    }

    #[test]
    fn misc_helpers_cover_boundaries() {
        assert_eq!(selected_tools(&[]), DEFAULT_TOOLS);
        assert_eq!(selected_tools(&["custom".to_string()]), vec!["custom"]);
        assert!(disk_available(Path::new(".")).unwrap() > 1024 * 1024 * 1024);
        let default_bin = default_bin_dir().unwrap();
        assert!(default_bin.ends_with(".local/bin"));
        assert!(default_bin.starts_with(home_dir().unwrap()));
        assert_eq!(format_bytes(7), "7B");
        assert_eq!(format_bytes(4096), "4096B");
        assert_eq!(format_bytes(1024 * 1024), "1.0MiB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.0MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.0MiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GiB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0GiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0GiB");

        let dir = tempfile::tempdir().unwrap();
        assert!(!is_managed_shim(&dir.path().join("missing")).unwrap());
        assert!(!is_managed_shim(dir.path()).unwrap());
        let shim = dir.path().join("shim");
        fs::write(&shim, format!("# {MANAGED_MARKER}\n")).unwrap();
        assert!(is_managed_shim(&shim).unwrap());
    }

    #[test]
    fn generated_path_shim_has_marker_and_env_escape() {
        let script = path_shim_script("vllm", Path::new("/tmp/infer-guard"), "24G", "4G");
        assert!(script.contains(MANAGED_MARKER));
        assert!(script.contains("INFER_GUARD_REAL_VLLM"));
        assert!(script.contains("INFER_GUARD_ALLOW_NO_EARLYOOM"));
    }

    #[test]
    fn real_path_appends_real_suffix() {
        assert_eq!(
            real_path_for(Path::new("/tmp/vllm")),
            PathBuf::from("/tmp/vllm.real")
        );
    }
}
