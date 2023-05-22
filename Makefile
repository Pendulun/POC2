INSTALL_DIR=install
_download_osmnx_whl:
	mkdir -p $(INSTALL_DIR)
	wget -P $(INSTALL_DIR) --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0" https://files.pythonhosted.org/packages/55/4f/748c11850ac1e79e3d30fdbabac2165dfbb9a2adda3fdc9807fafd0cd131/osmnx-1.3.0-py3-none-any.whl

install: _download_osmnx_whl
	pip install ./$(INSTALL_DIR)/*
	pip install -r requirements.txt