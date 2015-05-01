CC	=	g++ 
OBJS	=	main.o
SOURCES	=	main.cpp
LIBS    =       -lIL -pthread 
EXE	=	ale
FLAGS	=	-Wno-write-strings

$(EXE):	.
	$(CC) $(FLAGS) -c $(SOURCES) -o $(OBJS)
	$(CC) -o $(EXE) $(OBJS) $(LIBS)
