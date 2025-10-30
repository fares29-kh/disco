@echo off
REM Script pour lancer l'application avec Docker sur Windows

echo ========================================
echo  Process Mining Dashboard - Docker
echo ========================================
echo.

REM Vérifier si Docker est installé
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer Docker Desktop depuis: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [OK] Docker est installe
echo.

REM Vérifier si Docker est en cours d'exécution
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas en cours d'execution
    echo Veuillez demarrer Docker Desktop
    pause
    exit /b 1
)

echo [OK] Docker est en cours d'execution
echo.

REM Vérifier si docker-compose est disponible
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Utilisation de docker-compose...
    echo.
    
    REM Construire et lancer avec docker-compose
    echo [INFO] Construction de l'image Docker...
    docker-compose build
    
    echo.
    echo [INFO] Demarrage de l'application...
    docker-compose up -d
    
    echo.
    echo [OK] Application demarree avec succes!
    echo.
    echo ========================================
    echo  L'application est accessible a:
    echo  http://localhost:8501
    echo ========================================
    echo.
    echo Pour voir les logs: docker-compose logs -f
    echo Pour arreter: docker-compose down
    echo.
) else (
    echo [INFO] docker-compose non disponible, utilisation de docker run...
    echo.
    
    REM Construire l'image
    echo [INFO] Construction de l'image Docker...
    docker build -t process-mining-dashboard .
    
    echo.
    echo [INFO] Demarrage du conteneur...
    docker run -d ^
        --name process-mining-dashboard ^
        -p 8501:8501 ^
        --restart unless-stopped ^
        process-mining-dashboard
    
    echo.
    echo [OK] Application demarree avec succes!
    echo.
    echo ========================================
    echo  L'application est accessible a:
    echo  http://localhost:8501
    echo ========================================
    echo.
    echo Pour voir les logs: docker logs -f process-mining-dashboard
    echo Pour arreter: docker stop process-mining-dashboard
    echo.
)

REM Attendre quelques secondes
echo [INFO] Attente du demarrage de l'application (10 secondes)...
timeout /t 10 /nobreak >nul

REM Ouvrir le navigateur
echo [INFO] Ouverture du navigateur...
start http://localhost:8501

pause

