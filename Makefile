dat:
		lektor build --output-path $(HOME)/beaker-sites/ericmjl.com

gh_https:
		lektor deploy gh_https

gh_ssh:
		lektor deploy gh_ssh

do:
		lektor deploy digital_ocean
