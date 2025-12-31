@echo off
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VS_PATH=%%i
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0build"
cmake .. -DUSE_CUDA=OFF
cmake --build . --config Release
