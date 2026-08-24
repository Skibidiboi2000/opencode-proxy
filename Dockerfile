FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .

EXPOSE 5381

ENV OPENCODE_BASE_URL=https://opencode.ai/zen/v1
ENV OPENCODE_BROKE=false

ENTRYPOINT ["/entrypoint.sh"]
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
