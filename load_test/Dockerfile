FROM locustio/locust

WORKDIR /load_test
COPY locustfile.py .

CMD ["-f", "locustfile.py", "--host", "http://app:8000", "--users", "100", "--spawn-rate", "10", "--run-time", "5m"]
