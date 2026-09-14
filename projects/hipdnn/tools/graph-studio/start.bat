@echo off
setlocal
cd /d "%~dp0"

where bun >nul 2>&1 || (echo bun not found on PATH & exit /b 1)

if not exist node_modules (
  echo Installing dependencies...
  call bun install || exit /b 1
)

if /i "%~1"=="dev" (
  call bun run electron:dev
) else (
  call bun run electron:start
)
