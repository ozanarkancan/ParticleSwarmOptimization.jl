CFLAGS=-O3 -I.

libpso.so: psoupdate.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o libpso.so
