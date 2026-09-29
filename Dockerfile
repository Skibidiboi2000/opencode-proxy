FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py harness_tools.json ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5381

ENV OPENCODE_BASE_URL=https://opencode.ai/zen/v1

ENTRYPOINT ["/entrypoint.sh"]
