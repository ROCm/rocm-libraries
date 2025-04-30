help:
	@echo ""
	@echo "Demo on running tests with an external yaml file"
	@echo ""
	@echo "- help  : produces this help (default)"
	@echo "- run   : run the tests"
	@echo ""
	@echo "set the variable MATRICES_DIR (default is ./) to indicate where the matrix files are."
	@echo ""

MATRICES_DIR=./
ROCSPARSE_TEST_PATH=../../build/release/clients/staging
ROCSPARSE_TEST=$(ROCSPARSE_TEST_PATH)/rocsparse-test

run: demo.txt

%.txt:%.yaml
	$(ROCSPARSE_TEST) -I $(ROCSPARSE_TEST_PATH) --matrices-dir $(MATRICES_DIR) --yaml $< > $@ 2>&1
