all:
	g++ -std=c++11 -o vexclphold vexclphold.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system

clean:
	rm vexclphold.o
	rm vexclphold
