@echo off & setlocal

set "ROOT=%~dp0"
set "CMD=up"
set "REBUILD="
set "CONTAINER_NAME=trellis2"

rem =============================================
rem  1. Parse command-line arguments
rem =============================================
:parse
if "%~1"==""  goto :main
if /i "%~1"=="--rebuild" set "REBUILD=1" & shift & goto :parse
if /i "%~1"=="serve"   set "CMD=up"    & shift & goto :parse
if /i "%~1"=="build"   set "CMD=build" & shift & goto :parse
if /i "%~1"=="down"    set "CMD=down"  & shift & goto :parse
if /i "%~1"=="help"    goto :usage
shift & goto :parse

rem =============================================
rem  2. Main logic
rem =============================================
:main
cd /d "%ROOT%"

echo ============================================
echo  TRELLIS.2  ^|  docker compose  ^|  Mode: %CMD%
echo ============================================

rem -- build-only mode
if /i "%CMD%"=="build" (
    echo Building image via docker compose...
    docker compose build
    if errorlevel 1 goto :build_failed
    echo Done.
    exit /b 0
)

rem -- tear down mode
if /i "%CMD%"=="down" (
    echo Stopping and removing containers...
    docker compose down
    if errorlevel 1 exit /b 1
    exit /b 0
)

rem -- Explicit rebuild: rebuild the image and replace the container.
if defined REBUILD (
    echo Rebuilding image ^(--no-cache^) ...
    docker compose build --no-cache
    if errorlevel 1 goto :build_failed

    echo Recreating and starting services in the background...
    docker compose up -d --force-recreate
    if errorlevel 1 goto :start_failed
    call :wait_for_health
    if errorlevel 1 goto :health_failed
    echo Container is healthy. This window can now be closed.
    exit /b 0
)

rem -- Normal start: reuse the existing container without rebuilding/recreating it.
docker container inspect "%CONTAINER_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo Starting existing container "%CONTAINER_NAME%"...
    docker start "%CONTAINER_NAME%" >nul
    if errorlevel 1 goto :start_failed
    call :wait_for_health
    if errorlevel 1 goto :health_failed
    echo Container is healthy. This window can now be closed.
    exit /b 0
)

rem -- First run only: Compose creates the container and builds the image if needed.
echo Container "%CONTAINER_NAME%" does not exist; creating it in the background...
docker compose up -d
if errorlevel 1 goto :start_failed
call :wait_for_health
if errorlevel 1 goto :health_failed
echo Container is healthy. This window can now be closed.
exit /b 0

:wait_for_health
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%ROOT%docker_wait_healthy.ps1" -ContainerName "%CONTAINER_NAME%"
exit /b %ERRORLEVEL%

:build_failed
echo [ERROR] Image build failed.
exit /b 1

:start_failed
echo [ERROR] Container startup failed.
exit /b 1

:health_failed
echo [ERROR] Container did not become healthy. Check logs with: docker logs --timestamps %CONTAINER_NAME%
exit /b 1

rem =============================================
rem  3. Help
rem =============================================
:usage
echo.
echo  TRELLIS.2 Docker run script (docker compose)
echo  -------------------------------------------
echo  docker_run.bat [cmd] [--rebuild]
echo.
echo  Commands:
echo    serve       start/reuse container and wait until healthy (default)
echo    build       docker compose build - build image only
echo    down        docker compose down  - stop ^& remove containers
echo    --rebuild   force rebuild (--no-cache) and recreate the container
echo.
exit /b 0
