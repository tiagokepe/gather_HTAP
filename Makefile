CCFLAGS = -g -march=native -O3 -mavx512f 
TARGET=analytic_gather
LDFLAGS = 

all: gather column

column: CCFLAGS=$(CPPFLAGS)
column: src/analytic_column.c
	$(CC) $(CCFLAGS) -o analytic_column src/analytic_column.c

gather: operators.o utils.o analytic_gather.o hash_table.o vecmurmur.o
	$(CC) -o $(TARGET) utils.o operators.o analytic_gather.o hash_table.o vecmurmur.o $(LDFLAGS)

analytic_gather.o: src/analytic_gather.c
	$(CC) -c $(CCFLAGS) src/analytic_gather.c

utils.o: src/utils/utils.c
	$(CC) -c $(CCFLAGS) src/utils/utils.c

operators.o: src/operators.c src/hash_table.o src/vecmurmur.o
	$(CC) -c $(CCFLAGS) src/operators.c 

hash_table.o: src/hash_table.c
	$(CC) -c $(CCFLAGS) src/hash_table.c

vecmurmur.o: src/vecmurmur.c
	$(CC) -c $(CCFLAGS) src/vecmurmur.c


debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: gather column

clean:
	@rm -f *.o analytic_gather* analytic_column*
