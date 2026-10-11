@echo off
REM Double-click AFTER the files downloaded from Colab.
REM Keep this file in D:\Projects\FreshVision (next to app.py).
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "FOUND=0"

set "DET=%USERPROFILE%\Downloads\freshvision_detector.pt"
if exist "%DET%" (
  if not exist "models\det" mkdir "models\det"
  move /Y "%DET%" "models\det\best.pt"
  %PY% -c "from ultralytics import YOLO; m=YOLO(r'models\det\best.pt'); print('detector task:', m.task, '- classes:', len(m.names))"
  set "FOUND=1"
)

set "CLS=%USERPROFILE%\Downloads\freshvision_classifier.pt"
if exist "%CLS%" (
  if not exist "models\cls_new" mkdir "models\cls_new"
  move /Y "%CLS%" "models\cls_new\best.pt"
  %PY% -c "from ultralytics import YOLO; m=YOLO(r'models\cls_new\best.pt'); print('classifier task:', m.task, '- classes:', len(m.names))"
  set "FOUND=1"
)

if "%FOUND%"=="0" (
  echo Nothing found in %USERPROFILE%\Downloads
  echo Expected freshvision_detector.pt and/or freshvision_classifier.pt from Colab.
)
echo.
echo Detector must say "task: detect", classifier "task: classify". Then restart streamlit.
pause