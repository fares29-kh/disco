#!/bin/bash
# Script pour lancer l'application avec Docker sur Linux/Mac

echo "========================================"
echo " Process Mining Dashboard - Docker"
echo "========================================"
echo ""

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "[ERREUR] Docker n'est pas installé"
    echo "Veuillez installer Docker depuis: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "[OK] Docker est installé"
echo ""

# Vérifier si Docker est en cours d'exécution
if ! docker info &> /dev/null; then
    echo "[ERREUR] Docker n'est pas en cours d'exécution"
    echo "Veuillez démarrer Docker"
    exit 1
fi

echo "[OK] Docker est en cours d'exécution"
echo ""

# Vérifier si docker-compose est disponible
if command -v docker-compose &> /dev/null; then
    echo "[INFO] Utilisation de docker-compose..."
    echo ""
    
    # Construire et lancer avec docker-compose
    echo "[INFO] Construction de l'image Docker..."
    docker-compose build
    
    echo ""
    echo "[INFO] Démarrage de l'application..."
    docker-compose up -d
    
    echo ""
    echo "[OK] Application démarrée avec succès!"
    echo ""
    echo "========================================"
    echo " L'application est accessible à:"
    echo " http://localhost:8501"
    echo "========================================"
    echo ""
    echo "Pour voir les logs: docker-compose logs -f"
    echo "Pour arrêter: docker-compose down"
    echo ""
else
    echo "[INFO] docker-compose non disponible, utilisation de docker run..."
    echo ""
    
    # Construire l'image
    echo "[INFO] Construction de l'image Docker..."
    docker build -t process-mining-dashboard .
    
    echo ""
    echo "[INFO] Démarrage du conteneur..."
    docker run -d \
        --name process-mining-dashboard \
        -p 8501:8501 \
        --restart unless-stopped \
        process-mining-dashboard
    
    echo ""
    echo "[OK] Application démarrée avec succès!"
    echo ""
    echo "========================================"
    echo " L'application est accessible à:"
    echo " http://localhost:8501"
    echo "========================================"
    echo ""
    echo "Pour voir les logs: docker logs -f process-mining-dashboard"
    echo "Pour arrêter: docker stop process-mining-dashboard"
    echo ""
fi

# Attendre quelques secondes
echo "[INFO] Attente du démarrage de l'application (10 secondes)..."
sleep 10

# Ouvrir le navigateur (Linux/Mac)
if command -v xdg-open &> /dev/null; then
    echo "[INFO] Ouverture du navigateur..."
    xdg-open http://localhost:8501
elif command -v open &> /dev/null; then
    echo "[INFO] Ouverture du navigateur..."
    open http://localhost:8501
else
    echo "[INFO] Veuillez ouvrir manuellement: http://localhost:8501"
fi

echo ""
echo "Appuyez sur Entrée pour continuer..."
read

