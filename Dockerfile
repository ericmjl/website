FROM condaforge/mambaforge:23.1.0-4

WORKDIR /root
COPY blogbot /root/blogbot
RUN /opt/conda/bin/pip install -r /root/blogbot/requirements.txt

ENTRYPOINT ["uvicorn", "blogbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
