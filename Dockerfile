FROM dustynv/pytorch:2.7-r36.4.0

# Eviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Dépendances système nécessaires pour OpenCV (libxcb, X11, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-opencv \
        libxcb1 \
        libx11-6 \
        libx11-xcb1 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Forcer PyPI officiel (les index Jetson sont instables)
ENV PIP_INDEX_URL=https://pypi.org/simple

# Copier projet
WORKDIR /workspace
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --no-deps -r requirements.txt

COPY . .

# Par défaut, lance ton app
CMD ["python3", "main.py"]
