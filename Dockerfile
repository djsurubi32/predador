# 1. Puxa a imagem oficial do Python 3.11 super leve e estável
FROM python:3.11-slim

# 2. Define a pasta principal do servidor
WORKDIR /app

# 3. Instala os compiladores C e C++ do Linux necessários para XGBoost e CatBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia o seu arquivo de requisitos para dentro do servidor
COPY requirements.txt .

# 5. Instala as bibliotecas matemáticas do Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia todos os seus arquivos (.py, .env, etc) para o servidor
COPY . .

# 7. Dispara o ecossistema unificado 10/10
CMD ["python", "main.py"]
