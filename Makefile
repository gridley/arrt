main: arrt.C
	# g++ arrt.C -g -O0 -lgd -lstdc++fs -fstack-protector
	g++ arrt.C -O4 -ffast-math -lgd -lstdc++fs
