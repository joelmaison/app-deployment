@echo off
REM Batch script to evaluate all 5 systems
REM Make sure you've set CMU_API_KEY environment variable first!

echo ============================================================
echo Evaluating All 5 RAG Systems
echo ============================================================
echo.

REM Check if API key is set
if "%CMU_API_KEY%"=="" (
    echo ERROR: CMU_API_KEY environment variable not set!
    echo Please run: set CMU_API_KEY=your-api-key-here
    pause
    exit /b 1
)

echo.
echo [1/5] Evaluating System 1: None_gpt4omini
echo ============================================================
python src/evaluate.py --file output/prediction/None_gpt4omini.tsv
if %errorlevel% neq 0 (
    echo ERROR in System 1!
    pause
    exit /b 1
)

echo.
echo [2/5] Evaluating System 2: Azure_gpt4omini
echo ============================================================
python src/evaluate.py --file output/prediction/Azure_gpt4omini.tsv
if %errorlevel% neq 0 (
    echo ERROR in System 2!
    pause
    exit /b 1
)

echo.
echo [3/5] Evaluating System 3: Local_gpt4omini
echo ============================================================
python src/evaluate.py --file output/prediction/Local_gpt4omini.tsv
if %errorlevel% neq 0 (
    echo ERROR in System 3!
    pause
    exit /b 1
)

echo.
echo [4/5] Evaluating System 4: Azure_flant5
echo ============================================================
python src/evaluate.py --file output/prediction/Azure_flant5.tsv
if %errorlevel% neq 0 (
    echo ERROR in System 4!
    pause
    exit /b 1
)

echo.
echo [5/5] Evaluating System 5: Local_flant5
echo ============================================================
python src/evaluate.py --file output/prediction/Local_flant5.tsv
if %errorlevel% neq 0 (
    echo ERROR in System 5!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ ALL EVALUATIONS COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Evaluation files are in: output\evaluation\
echo.

pause
