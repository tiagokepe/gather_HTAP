CCFLAGS = -march=native
TARGET=analytic_release

all: utils.o analytic_workload.o
	$(CC) -o $(TARGET) utils.o analytic_workload.o

analytic_workload.o: analytic_workload.c
	$(CC) -c $(CCFLAGS) analytic_workload.c

utils.o: utils/utils.c
	$(CC) -c $(CCFLAGS) utils/utils.c

debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: TARGET=analytic_debug
debug: all

clean:
	@rm -f *.o analytic_release analytic_debug
