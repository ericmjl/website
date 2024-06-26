.PHONY: blogbot website

blogbot:
		panel serve apis/blogbot/app.ipynb --address 0.0.0.0 --allow-websocket-origin=0.0.0.0:5006 --dev

website:
		lektor server --port 5959 --host 0.0.0.0
