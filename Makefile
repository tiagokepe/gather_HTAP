CCFLAGS = -march=native
TARGET=analytic_gather

all: utils.o analytic_workload.o
	$(CC) -o $(TARGET) utils.o analytic_workload.o

analytic_workload.o: src/analytic_workload.c
	$(CC) -c $(CCFLAGS) src/analytic_workload.c

utils.o: src/utils/utils.c
	$(CC) -c $(CCFLAGS) src/utils/utils.c

debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: TARGET=analytic_gather_debug
debug: all

clean:
	@rm -f *.o analytic_gather*
