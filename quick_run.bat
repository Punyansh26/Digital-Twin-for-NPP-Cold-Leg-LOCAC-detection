@echo off
REM Quick run script for AP1000 Digital Twin
REM This runs the complete pipeline with mock data

echo ======================================================================
echo AP1000 Digital Twin - Quick Run
echo ======================================================================
echo.
echo This will run the complete pipeline with mock data
echo Estimated time: 5-10 minutes on GPU
echo.
pause

python run_pipeline.py --use-mock-data

echo.
echo ======================================================================
echo Pipeline Complete!
echo ======================================================================
echo.
echo View results in:
echo   results\plots\
echo   results\models\
echo.
pause
