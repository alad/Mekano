tests: build/lib.macosx-10.4-universal-2.6/mekano
	cd tests && nosetests -v
	@touch tests

build/lib.macosx-10.4-universal-2.6/mekano: FORCE
	python setup.py build
	sudo python setup.py install

dist: tests
	python setup.py sdist
	ssh mh "rm -rf /tmp/mekano*"
	scp dist/`ls -t dist | head -n 1` mh:/tmp/
	ssh mh /usr0/alad/install_mekano.sh
	
doc: tests
	cd /Users/alad/ && epydoc -v --html -o /Users/alad/src/mekano/docs mekano

clean:
	rm -rf build/
	sudo rm -rf /Library/Python/2.6/site-packages/mekano*
	rm -rf /tmp/tmp
	rm -rf /tmp/pyrex
	
timing: tests
	cd timing && python avtime.py
	
FORCE:

