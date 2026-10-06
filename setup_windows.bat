@echo off
setlocal

echo Creating Python virtual environment...
py -m venv .venv
if errorlevel 1 (
    echo Failed to create .venv. Check that Python is installed and the 'py' command works.
    exit /b 1
)

echo.
echo Virtual environment created.
echo To activate it in PowerShell:
echo   .venv\Scripts\Activate.ps1
echo.
echo Do not install project dependencies yet; requirements.txt will be pinned after we inspect the official FloodPlanet repository.
endlocal
