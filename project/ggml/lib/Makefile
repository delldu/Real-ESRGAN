#/************************************************************************************
#***
#***	Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Tue 30 Jan 2024 11:53:11 PM CST
#***
#************************************************************************************/
#

LIBNAME := libggml_engine
PKGCONFNAME :=ggml_engine.pc

ENGINE_VERSION := V1.0.0
RELEASE_DATE := $(shell date '+%Y-%m-%d')

# Installation related variables and target
PREFIX?=/usr/local
INCLUDE_PATH?=include/ggml_engine
LIBRARY_PATH?=lib
PKGCONF_PATH?=pkgconfig
INSTALL_INCLUDE_PATH= $(PREFIX)/$(INCLUDE_PATH)
INSTALL_LIBRARY_PATH= $(PREFIX)/$(LIBRARY_PATH)
INSTALL_PKGCONF_PATH= $(INSTALL_LIBRARY_PATH)/$(PKGCONF_PATH)

INCS	:=  \
		$(shell pkg-config nimage --cflags) \
		$(shell pkg-config ggml --cflags) \
		-Ilib/include

SOURCE :=  \
	source/ggml_engine.c

DEFINES := -DGGML_CUBLAS -DENGINE_VERSION="\"${ENGINE_VERSION}\"" -DRELEASE_DATE="\"${RELEASE_DATE}\"" \
 	-D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11 -std=gnu++11

CFLAGS := -O3 -fPIC -Wall -Wextra
CXXFLAGS := ${CFLAGS}
LDFLAGS := -fPIC


#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBNAME).a $(OBJECTS)


#****************************************************************************
# Depend block
#****************************************************************************
depend:

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


clean:
	rm -rf *.a *.so *.o $(OBJECTS)

install:
	sudo mkdir -p ${INSTALL_INCLUDE_PATH}
	sudo cp include/*.h ${INSTALL_INCLUDE_PATH}
	sudo cp ${LIBNAME}.a ${INSTALL_LIBRARY_PATH}

	sudo mkdir -p ${INSTALL_PKGCONF_PATH}
	sudo cp ${PKGCONFNAME} ${INSTALL_PKGCONF_PATH}


#cmake -DGGML_CUBLAS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
