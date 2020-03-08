# --------------------------------------------------------------------
# Copyright (c) 2019 LINKIT, The Netherlands. All Rights Reserved.
# Author(s): Anthony Potappel
# 
# This software may be modified and distributed under the terms of the
# MIT license. See the LICENSE file for details.
# --------------------------------------------------------------------
# There should be no need to change this,
# if you do you'll also need to update docker-compose.yml
SERVICE_TARGET := pyshipper

# If you see pwd_unknown showing up, this is why. Re-calibrate your system.
PWD ?= pwd_unknown

# retrieve NAME from /variables file
MODULE_NAME = \
	$(shell awk -F= '/^NAME\ ?=/{gsub(/\47|"/, "", $$NF);print $$NF;exit}' variables)
MODULE_VERSION = \
	$(shell awk -F= '/^VERSION\ ?=/{gsub(/\47|"/, "", $$NF);print $$NF;exit}' variables)

# if vars not set specifially: try default to environment, else fixed value.
# strip to ensure spaces are removed in future editorial mistakes.
# tested to work consistently on popular Linux flavors and Mac.
ifeq ($(user),)
# USER retrieved from env, UID from shell.
HOST_USER ?= $(strip $(if $(USER),$(USER),nodummy))
HOST_UID ?= $(strip $(if $(shell id -u),$(shell id -u),4000))
else
# allow override by adding user= and/ or uid=  (lowercase!).
# uid= defaults to 0 if user= set (i.e. root).
HOST_USER = $(user)
HOST_UID = $(strip $(if $(uid),$(uid),0))
endif

# cli prefix for commands to run in container
RUN_DOCK = \
	docker-compose -p $(MODULE_NAME)_$(HOST_UID) run --rm $(SERVICE_TARGET) sh -l -c

# export such that its passed to shell functions for Docker to pick up.
export MODULE_NAME
export HOST_USER
export HOST_UID



.PHONY: shell 
shell:
	$(RUN_DOCK) "cd $(MODULE_NAME) \
		&& ([ -d "$(MODULE_NAME)" ] || ln -sf module "$(MODULE_NAME)") \
		&& bash"

.PHONY: fork
FILES = \
	module files Makefile Dockerfile docker-compose.yml setup.py MANIFEST.in variables .gitignore .pylintrc
DEST ?= $(dest)
fork:
ifeq ($(DEST),)
	@echo 'MISSING "dest=" PARAMETER'
	@echo 'RUN: make fork dest=$${DIRECTORY}'
else
	@# copy when either directory does not exist, or is empty
	@# skip if a non-empty directory exist
	@# Note this excludes README.md and LICENSE -- you own your own project ;)
	@(([ ! -d "$(DEST)" ] || find "$(DEST)" -prune -type d -empty |grep -q .) \
	  && mkdir -p "$(DEST)" \
      && cp -R $(FILES) "$(DEST)/" \
	  && touch $(DEST)/README.md \
      || echo "SKIPPING FORK, WON'T OVERWRITE EXISTING DIRECTORY")
endif

.PHONY: module
module: 
	@# ensure there is a symlink from MODULE_NAME to module directory
	@# then run regular setup.py to build the module
	$(RUN_DOCK) "cd $(MODULE_NAME) \
		&& find ./ -type l -maxdepth 1 |xargs rm -f \
		&& ln -sf module "$(MODULE_NAME)" \
		&& python3 setup.py sdist"

.PHONY: pylint
pylint:
	$(RUN_DOCK) "cd $(MODULE_NAME)/module \
		&& pylint --rcfile=../.pylintrc * -f parseable"

.PHONY: upload
upload:
	$(RUN_DOCK) "twine upload $(MODULE_NAME)/dist/$(MODULE_NAME)-$(MODULE_VERSION)*"

.PHONY: clean
clean:
	$(RUN_DOCK) "cd $(MODULE_NAME) \
		&& rm -rf ./build ./dist ./*.egg-info \
		&& find ./ -type l -maxdepth 1 |xargs rm -f \
		&& find ./$(MODULE) -type d -name '__pycache__' |xargs rm -rf"
