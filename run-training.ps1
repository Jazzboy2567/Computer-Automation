# Launch the SPD demo-seeded RL training as a LOW-IMPACT background job.
#
# Caps the whole training tree (the python worker AND the java EnvServers it spawns)
# to a few efficiency cores at Below-Normal priority, so it never competes with your
# foreground apps (Vivaldi, editors, etc.). Progress is unaffected by the cap; it just
# runs on the leftover cores.
#
#   Usage:  powershell -ExecutionPolicy Bypass -File .\run-training.ps1
#
# Stop it later with:  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
#   ? { $_.CommandLine -match 'pilot rl' } | % { Stop-Process -Id $_.ProcessId -Force }

param(
  # logical CPUs to confine training to. Default = ALL 8 efficiency-core threads on a
  # Core Ultra 155H (12-19). Pinning python AND the java EnvServers to only 4 shared
  # cores starved them into a livelock (python spun, JVMs couldn't answer, 0 progress),
  # so give the whole tree the full E-core block — still off the P-cores your foreground
  # apps use, but with enough headroom to actually run. Adjust if your CPU layout differs.
  [int[]] $Cores = @(12,13,14,15,16,17,18,19),
  [string] $Log  = 'spd_goo_demo2.log',
  [string] $Err  = 'spd_goo_demo2.err'
)

$ErrorActionPreference = 'Stop'
$proj = $PSScriptRoot
$mask = [IntPtr]([int]([int64](($Cores | ForEach-Object { 1 -shl $_ }) -join '+' | Invoke-Expression)))

# Single-threaded BLAS. numpy's multithreaded BLAS (OpenBLAS) spin-waits to sync its
# worker threads; once we pin this process to a few cores, those spinning threads
# livelock on the shared cores — the worker burns CPU on thread-sync and never steps
# the env (the "hang"). Our net is tiny, so 1 thread is just as fast and removes the spin.
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$env:PILOT_SEED_ACQ_KIT_PROB = '1.0'   # gear-up materials on floor 1
$env:PILOT_DENSE_MAP = '1'             # dense tile map (matches the demos)
$env:PILOT_DQN_BUFFER = '15000'        # smaller replay buffer: the dense map makes obs
                                       # ~1014 floats, so 50k would be ~400MB and swap on
                                       # a RAM-tight machine (-> the forward pass crawls).
                                       # 15k ~= 120MB, still ample with demos always seeded.

$p = Start-Process -FilePath "$proj\.venv\Scripts\python.exe" `
  -ArgumentList '-m','pilot','rl','--game','spd-real','--agent','dqn',
                '--episodes','4000','--curriculum','5','--decay','30000',
                '--max-steps','300','--forever','--demos','goo_demos.joblib','--demo-frac','0.05' `
  -WorkingDirectory $proj `
  -RedirectStandardOutput "$proj\$Log" -RedirectStandardError "$proj\$Err" `
  -WindowStyle Hidden -PassThru

Write-Host "launched training pid $($p.Id); capping to cores $($Cores -join ',') at BelowNormal"

# Cap the launcher/worker immediately; then for ~90s catch the java EnvServers it
# spawns (and the child worker python) and cap those too. New children spawned later
# (e.g. a crash-rebuild) inherit the parent's affinity, so this covers respawns.
$deadline = (Get-Date).AddSeconds(90)
$seen = @{}
while ((Get-Date) -lt $deadline) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='java.exe'" |
    Where-Object { $_.CommandLine -match 'pilot rl' -or $_.CommandLine -match 'rlbridge' } |
    ForEach-Object {
      if (-not $seen.ContainsKey($_.ProcessId)) {
        try {
          $pr = Get-Process -Id $_.ProcessId
          $pr.ProcessorAffinity = $mask
          $pr.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
          $seen[$_.ProcessId] = $true
        } catch {}
      }
    }
  Start-Sleep -Milliseconds 1500
}
Write-Host "capped $($seen.Count) training processes: $($seen.Keys -join ', ')"
