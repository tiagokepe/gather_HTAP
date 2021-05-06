CCFLAGS = -march=native
TARGET=analytic_gather

all: gather column

column: CCFLAGS=$(CPPFLAFS)
column: src/analytic_column.c
	$(CC) -o $(CCFLAGS) analytic_column src/analytic_column.c

gather: utils.o analytic_gather.o
	$(CC) -o $(TARGET) utils.o analytic_gather.o

analytic_gather.o: src/analytic_gather.c
	$(CC) -c $(CCFLAGS) src/analytic_gather.c

utils.o: src/utils/utils.c
	$(CC) -c $(CCFLAGS) src/utils/utils.c

debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: TARGET=analytic_gather_debug
debug: gather column

clean:
	@rm -f *.o analytic_gather* analytic_column*
