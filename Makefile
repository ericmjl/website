.PHONY: blogbot website

blogbot:
		uvicorn blogbot.api:app --reload

website:
		lektor server --port 5959 --host 0.0.0.0
