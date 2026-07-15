# sage-bioclip2 -- Makefile
#
# `make test` runs the OFFLINE unit suite: pure-stdlib cache-consumer logic
# (consumer / selection / seenstore / identity) plus the bioclip2-specific tests
# as they land. Self-bootstraps a throwaway venv so it runs on a clean checkout.
#
# The GPU integration test (real BioCLIP2 inference) is separate.

VENV := .venv-test
PY   := $(VENV)/bin/python

.PHONY: test clean

test: $(VENV)/.stamp
	$(PY) -m pytest -q tests/test_consumer.py tests/test_consumer_meta.py tests/test_selection.py tests/test_seenstore.py tests/test_identity.py tests/test_app_bioclip.py

$(VENV)/.stamp:
	python3 -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet pytest Pillow piexif numpy
	touch $@

clean:
	rm -rf $(VENV)
