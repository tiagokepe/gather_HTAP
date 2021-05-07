CCFLAGS = -march=native -O3 -mavx512f
TARGET=analytic_gather

all: gather column

column: CCFLAGS=$(CPPFLAGS)
column: src/analytic_column.c
	$(CC) $(CCFLAGS) -o analytic_column src/analytic_column.c

gather: utils.o analytic_gather.o
	$(CC) -o $(TARGET) utils.o analytic_gather.o

analytic_gather.o: src/analytic_gather.c
	$(CC) -c $(CCFLAGS) src/analytic_gather.c

utils.o: src/utils/utils.c
	$(CC) -c $(CCFLAGS) src/utils/utils.c

debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: gather column

clean:
	@rm -f *.o analytic_gather* analytic_column*
