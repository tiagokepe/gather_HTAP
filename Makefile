CCFLAGS = -g -march=native -O3 -mavx512f -Wall
TARGET1=analytic_column
TARGET2=analytic_gather
LDFLAGS = -lm

all: gather column

column: col_operators.o utils.o analytic_column.o hash_table.o vecmurmur.o
	$(CC) -o $(TARGET1) utils.o col_operators.o analytic_column.o hash_table.o vecmurmur.o $(LDFLAGS)

gather: operators.o utils.o analytic_gather.o hash_table.o vecmurmur.o
	$(CC) -o $(TARGET2) utils.o operators.o analytic_gather.o hash_table.o vecmurmur.o $(LDFLAGS)

analytic_gather.o: src/analytic_gather.c
	$(CC) -c $(CCFLAGS) src/analytic_gather.c $(LDFLAGS)

analytic_column.o: src/analytic_column.c
	$(CC) -c $(CCFLAGS) src/analytic_column.c $(LDFLAGS)

utils.o: src/utils/utils.c
	$(CC) -c $(CCFLAGS) src/utils/utils.c $(LDFLAGS)

col_operators.o: src/col_operators.c src/hash_table.o src/vecmurmur.o
	$(CC) -c $(CCFLAGS) src/col_operators.c $(LDFLAGS)

operators.o: src/operators.c src/hash_table.o src/vecmurmur.o
	$(CC) -c $(CCFLAGS) src/operators.c $(LDFLAGS)

hash_table.o: src/hash_table.c
	$(CC) -c $(CCFLAGS) src/hash_table.c $(LDFLAGS)

vecmurmur.o: src/vecmurmur.c
	$(CC) -c $(CCFLAGS) src/vecmurmur.c


debug: CPPFLAGS = -DDEBUG
debug: CCFLAGS += $(CPPFLAGS)
debug: gather column

clean:
	@rm -f *.o analytic_gather* analytic_column*
