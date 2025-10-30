# Dockerfile pour Process Mining Dashboard
# Image de base Python officielle
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Process Mining Team"
LABEL description="Process Mining & AI Dashboard for Bug Workflow Analysis"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
# graphviz est nécessaire pour la visualisation des process maps
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    graphviz-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

# Changer vers l'utilisateur non-root
USER streamlit

# Exposer le port Streamlit
EXPOSE 8501

# Health check pour vérifier que l'application fonctionne
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

