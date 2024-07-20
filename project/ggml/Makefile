#
#/************************************************************************************
#***
#***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Wed 17 Jan 2024 10:36:30 PM CST
#***
#************************************************************************************/
#

TARGET := image_zoom4x
SOURCE := main.cpp

RELEASE_DATE := $(shell date '+%Y-%m-%d')

INCS	:=  \
		-Ilib/include \
		$(shell pkg-config nimage --cflags) \
		$(shell pkg-config ggml --cflags) 

LIBS	:= \
	$(shell pkg-config nimage --libs) \
	$(shell pkg-config ggml --libs) -lgomp \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lpthread -lm -lstdc++

CXXFLAGS := -O2 -fPIC
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
DEFINES  := -DGGML_CUDA -DRELEASE_DATE="\"${RELEASE_DATE}\"" \
		-D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11 -std=gnu++11

CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

all: ${TARGET}
# install

#****************************************************************************
# Compile block
#****************************************************************************
${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}

format:
	clang-format --style="{BasedOnStyle: WebKit, TabWidth: 4, IndentWidth: 4, ColumnLimit: 120}" \
	-i main.cpp rrdbnet.h
