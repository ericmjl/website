.PHONY: blogbot

dat:
		lektor build --output-path $(HOME)/beaker-sites/ericmjl.com

gh_https:
		lektor deploy gh_https

gh_ssh:
		lektor deploy gh_ssh

do:
		lektor deploy digital_ocean

update:
		wget https://raw.githubusercontent.com/ericmjl/conda-envs/master/lektor.yml -O environment.yml
		conda env update -f environment.yml

blogbot:
		uvicorn blogbot.api:app --reload
