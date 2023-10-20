.PHONY: blogbot website

blogbot:
		uvicorn apis.blogbot.api:app --reload --host 0.0.0.0 --port 8003

website:
		lektor server --port 5959 --host 0.0.0.0
