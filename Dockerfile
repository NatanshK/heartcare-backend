FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

WORKDIR /app


COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 7860 

CMD ["/bin/sh", "-c", "exec gunicorn app:app --bind \"0.0.0.0:${PORT:-7860}\" --worker-class sync --workers 2 --timeout 120 --log-level debug --access-logfile - --error-logfile -"]
