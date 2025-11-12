GROQ_DIR ?= /opt/groq/runtime.0.9.3

CC = g++

# define any compile-time flags
CFLAGS = -Wall -fPIC

# define any directories containing header files other than /usr/include
INCLUDES = -I$(GROQ_DIR)/include

# define library paths in addition to /usr/lib
LFLAGS =

# define any libraries to link into executable:
LIBS = $(GROQ_DIR)/lib/runtimelib/lib/libgroqruntime.a -lrt -pthread -ldl

# define the C source files
SRCS = qcsim_runtime_main.c

geniop:
	mkdir -p build
#    python groqQCsim.py
#	for FILE in usiop/us*.0.iop; do xxd -i $$FILE $$FILE.h; done

svall:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/libsvDFE.so $(SRCS) $(LFLAGS) $(LIBS) -shared -O3 -fPIC -Wall

testsv:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testsv $(SRCS) $(LFLAGS) $(LIBS) -DTEST -O3 -Wall -g

all: geniop svall

test: geniop testsv

# DO NOT DELETE THIS LINE -- make depend needs it
