install-req:
	pip install -r requirements.txt --index-url http://nexus.prod.uci.cu/repository/pypi-proxy/simple/ --trusted-host nexus.prod.uci.cu

install-req-dev:
	pip install -r requirements.txt -r requirements-dev.txt --index-url http://nexus.prod.uci.cu/repository/pypi-proxy/simple/ --trusted-host nexus.prod.uci.cu

test:
	pytest -v

setup:
	python setup.py sdist bdist_wheel
	twine check dist/*$(v)*

publish:
	twine upload dist/*$(v)*
