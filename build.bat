@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

echo ============================================================
echo  SimpliSQL - EXE Build
echo ============================================================
echo.

set "PYTHON=%SCRIPT_DIR%..\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [1/5] Using Python:
echo        %PYTHON%
echo.

"%PYTHON%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found. Installing...
    "%PYTHON%" -m pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Could not install PyInstaller.
        goto :fail
    )
)

set "DIST_DIR=%SCRIPT_DIR%dist_clickbuild"
set "WORK_DIR=%SCRIPT_DIR%build_clickbuild"

echo [2/5] Cleaning previous click-build folders...
if exist "%WORK_DIR%" rmdir /s /q "%WORK_DIR%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
echo.

echo [3/5] Removing __pycache__ folders...
for /d /r "%SCRIPT_DIR%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)
echo.

echo [4/5] Building EXE with PyInstaller...
echo        Spec file : SimpliSQL.spec
echo        Dist path : %DIST_DIR%
echo        Work path : %WORK_DIR%
echo.
"%PYTHON%" -m PyInstaller SimpliSQL.spec --noconfirm --distpath "%DIST_DIR%" --workpath "%WORK_DIR%"
if errorlevel 1 (
    echo.
    echo [ERROR] EXE build failed.
    goto :fail
)

echo.
echo [5/5] Build complete.
echo.
echo  EXE path:
echo  %DIST_DIR%\SimpliSQL\SimpliSQL.exe
echo.
if exist "%DIST_DIR%\SimpliSQL\SimpliSQL.exe" (
    start "" explorer "%DIST_DIR%\SimpliSQL"
)

popd >nul
pause
exit /b 0

:fail
echo.
echo Build did not finish successfully.
popd >nul
pause
exit /b 1
