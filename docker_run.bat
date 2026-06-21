@echo off & setlocal

set "ROOT=%~dp0"
set "IMG=trellis2:latest"
set "NAME=trellis2"
set "CMD=demo"
set "REBUILD="

rem =============================================
rem  1. Parse command-line arguments
rem =============================================
:parse
if "%~1"==""  goto :main
if /i "%~1"=="--rebuild" set "REBUILD=1" & shift & goto :parse
for %%C in (demo app texturing bash build) do if /i "%~1"=="%%C" set "CMD=%%C"
if /i "%~1"=="help" goto :usage
shift & goto :parse

rem =============================================
rem  2. Build image (shared subroutine)
rem =============================================
:build
docker build -t %IMG% -f "%ROOT%Dockerfile" "%ROOT%"
docker images -q %IMG% 2>nul | findstr . >nul || (
    echo [ERROR] Build failed! & pause & exit /b 1
)
exit /b

rem =============================================
rem  3. Main logic
rem =============================================
:main
echo ============================================
echo  TRELLIS.2  ^|  Mode: %CMD%
echo ============================================

rem -- Check / build image
docker images -q %IMG% 2>nul | findstr . >nul
if errorlevel 1 (
    echo Building image...
    call :build
) else if defined REBUILD (
    echo Rebuilding image per --rebuild...
    call :build
) else (
    echo Image exists, skipping build.
)

if /i "%CMD%"=="build" (echo Done. & pause & exit /b 0)

rem -- Clean & launch
docker stop %NAME% >nul 2>&1
docker rm   %NAME% >nul 2>&1

echo Starting [%CMD%] ...
docker run -it --rm ^
    --name %NAME% --gpus all ^
    -e NVIDIA_VISIBLE_DEVICES=all ^
    -e OPENCV_IO_ENABLE_OPENEXR=1 ^
    -e ATTN_BACKEND=flash-attn ^
    -e HF_HOME=/workspace/.cache/huggingface ^
    -v "%ROOT%:/workspace/TRELLIS.2" ^
    -v models:/models:ro ^
    -v hf_cache:/workspace/.cache/huggingface ^
    -p 7860:7860 -w /workspace/TRELLIS.2 ^
    %IMG% %CMD%

if errorlevel 1 (echo [ERROR] Container exited with code %ERRORLEVEL% & pause)
exit /b %ERRORLEVEL%

rem =============================================
rem  4. Help
rem =============================================
:usage
echo.
echo  TRELLIS.2 Docker run script
echo  ------------------------------
echo  docker_run.bat [cmd] [--rebuild]
echo.
echo  Commands:
echo    demo        image-to-3D demo (default)
echo    app         Gradio UI ^(http://localhost:7860^)
echo    texturing   texturing demo
echo    bash        interactive shell
echo    build       build image only
echo    --rebuild   force rebuild image
echo.
pause & exit /b 0
