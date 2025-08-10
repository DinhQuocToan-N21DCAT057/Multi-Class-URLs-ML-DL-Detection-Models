@echo off
echo Clearing Python cache files...
if exist "app\__pycache__" rmdir /s /q "app\__pycache__"
if exist "scripts\__pycache__" rmdir /s /q "scripts\__pycache__"
if exist "configs\__pycache__" rmdir /s /q "configs\__pycache__"
if exist "utils\__pycache__" rmdir /s /q "utils\__pycache__"
echo Cache cleared successfully!
echo.
echo Starting Flask application...
python run.py
