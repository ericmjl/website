.PHONY: blogbot website

blogbot:
		BOKEH_ALLOW_WS_ORIGIN=0.0.0.0:5006 panel serve apis/blogbot/app.ipynb --address 0.0.0.0

website:
		lektor server --port 5959 --host 0.0.0.0
