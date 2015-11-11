int:
	g++ -std=c++11 -o vexclphold_int vexclphold_int.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system

float:
	g++ -std=c++11 -o vexclphold_float vexclphold_float.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system

clean:
	rm vexclphold_int
	rm vexclphold_float
