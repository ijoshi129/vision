<#
.SYNOPSIS
    Set up Vision on Windows: uv, a Python 3.12 virtualenv in the checkout, the dependencies and,
    with -Voice, the speech models.

.DESCRIPTION
    Needs no administrator rights and installs nothing system-wide (uv, if missing, goes in for your
    user via winget, after asking). Safe to re-run: an existing .venv is reused and nothing already
    installed is removed.

        powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 [-Voice] [-Serve] [-All] [-AddToPath] [-Yes]

.PARAMETER Voice
    Speech in and out: the voice extra (CUDA torch, faster-whisper, Qwen3-TTS; about 6 GB), then
    `vision setup` for the speech models (about 10 GB, into %USERPROFILE%\.cache\huggingface).
.PARAMETER Serve
    `vision serve`, for the Vision Remote iPhone app.
.PARAMETER All
    Voice, serve and weather.
.PARAMETER AddToPath
    Put the checkout's bin folder on your user PATH, so `vision` works in any new terminal.
.PARAMETER Yes
    Don't ask before installing uv or downloading the voice packages and models.
#>
[CmdletBinding()]
param(
    [switch]$Voice,
    [switch]$Serve,
    [switch]$All,
    [switch]$AddToPath,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Say([string]$Message) { Write-Host "vision setup: $Message" }

function Confirm-Step([string]$Question) {
    if ($Yes) { return $true }
    $answer = Read-Host "$Question [y/N]"
    return $answer -match '^(y|yes)$'
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed (exit code $LASTEXITCODE)" }
}

# Git for Windows: Claude Code's Bash tool and the local model's shell run its bash.exe.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Say 'Git for Windows is required: https://git-scm.com/download/win (or: winget install Git.Git)'
    exit 1
}

# Claude Code must be the native claude.exe: cmd.exe (which runs npm's claude.cmd) cuts Vision's
# multi-line system prompt at its first newline and drops every flag after it.
$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
    Say 'Claude Code is not installed. Install it with:  irm https://claude.ai/install.ps1 | iex'
    Say 'then run `claude` once to log in. (Vision can also use Codex, Grok or a local model.)'
} elseif ($claude.Source -match '\.(cmd|bat)$') {
    Say "Claude Code was found as $($claude.Source), npm's launcher. Vision needs the native claude.exe:"
    Say '  irm https://claude.ai/install.ps1 | iex'
}

# uv. A terminal opened before uv was installed still has the old PATH, so look at the saved PATH and
# at winget's package folder too before deciding it's missing.
function Find-Uv {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    if (-not (Confirm-Step 'uv (the Python package manager Vision uses) is not installed. Install it for your user with winget?')) {
        Say 'uv is required: https://docs.astral.sh/uv/getting-started/installation/'
        exit 1
    }
    # winget's exit code also flags harmless outcomes ("already installed"), so judge by whether uv is there after.
    & winget install --id astral-sh.uv -e --scope user --accept-source-agreements --accept-package-agreements
    $uv = Find-Uv
    if (-not $uv) { Say 'uv could not be installed with winget: see https://docs.astral.sh/uv/getting-started/installation/'; exit 1 }
}
Say "using uv at $uv"

$extras = @()
if ($Voice -or $All) { $extras += 'voice' }
if ($Serve -or $All) { $extras += 'serve' }
if ($All) { $extras += 'weather' }

if ($extras -contains 'voice') {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Say 'No NVIDIA GPU found (nvidia-smi): the voice will run on the CPU, several times slower than real time.'
    }
    if (-not (Confirm-Step 'Voice downloads about 6 GB of packages now and about 10 GB of speech models after. Continue?')) {
        Say 'Skipping voice; run this again with -Voice when you want it.'
        $extras = @($extras | Where-Object { $_ -ne 'voice' })
    }
}

Push-Location $root
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        Say 'creating .venv (Python 3.12; uv downloads it if needed)'
        Invoke-Checked $uv @('venv', '--python', '3.12', '.venv')
    }
    $sync = @('sync', '--frozen', '--inexact')  # --inexact: a re-run never removes what an earlier one installed
    foreach ($extra in $extras) { $sync += @('--extra', $extra) }
    Say ("installing: text chat" + ($(if ($extras) { ' + ' + ($extras -join ', ') } else { '' })))
    Invoke-Checked $uv $sync

    $env:PYTHONUTF8 = '1'
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if ($extras -contains 'voice') {
        Say 'fetching the speech models (vision setup)'
        Invoke-Checked $python @('-m', 'vision', 'setup')
    }

    if ($AddToPath) {
        $bin = Join-Path $root 'bin'
        # Straight to the registry, so the existing entries keep their exact text and value type.
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        $current = $key.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
        $kind = if ($current) { $key.GetValueKind('Path') } else { [Microsoft.Win32.RegistryValueKind]::ExpandString }
        if (($current -split ';') -contains $bin) {
            Say "$bin is already on your PATH"
        } else {
            $new = if (-not $current) { $bin } elseif ($current.EndsWith(';')) { "$current$bin" } else { "$current;$bin" }
            $key.SetValue('Path', $new, $kind)
            # Tell running programs (Explorer, new terminals) that the environment changed.
            [Environment]::SetEnvironmentVariable('VISION_SETUP_REFRESH', '1', 'User')
            [Environment]::SetEnvironmentVariable('VISION_SETUP_REFRESH', $null, 'User')
            Say "added $bin to your user PATH (new terminals pick it up)"
        }
        $key.Close()
    }

    if ($claude -and $claude.Source -notmatch '\.(cmd|bat)$') {
        Say 'checking the install (vision doctor: one small Claude request)'
        & $python -m vision doctor --no-usage
    }
} finally {
    Pop-Location
}

$start = if ($AddToPath) { 'vision' } else { Join-Path $root 'bin\vision.cmd' }
Say "done. Start a chat with:  $start      (see WINDOWS.md for what works on Windows)"
