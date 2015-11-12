vex:
	g++ -std=c++11 -o vexclphold vexclphold_float.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system
	
airport:
	g++ -std=c++11 -o vexclarpt vexclarpt.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system
	
boost:
	g++ -std=c++11 -o boostclphold boostclphold.cpp -I./ -I./CL -I/usr/local/include/compute/ -L/usr/lib/x86_64-linux-gnu/ -lOpenCL
	
vex_int:
	g++ -std=c++11 -o vexclphold_int vexclphold_int.cpp -I. -I./vexcl -I./CL -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lboost_system

clean:
	rm -f vexclphold_int
	rm -f vexclphold_float
	rm -f vexclarpt
	rm -f boostclphold
	rm -f *.*~
	rm -f *~
