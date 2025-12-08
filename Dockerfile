FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Instalar torchaudio y libs de audio
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      soundfile \
      librosa \
      torchmetrics[audio] \
      asteroid \
      PyYAML \
      pystoi \
      pesq \
      datasets


# Directorio de trabajo
WORKDIR /workspace

# Copiar el proyecto
COPY . /workspace

# Asegurar carpetas (aunque main.py también las crea)
RUN mkdir -p /workspace/data /workspace/results

# Comando por defecto: entrenar
CMD ["python", "main.py"]
