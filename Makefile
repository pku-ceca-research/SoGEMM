VERSION=1.0.0
DATE=$(shell date +'%y%m%d_%H%M%S')
NAME=

# Load config file for each different build
CONFIG_FILE := Makefile.config
ifeq ($(wildcard $(CONFIG_FILE)),)
	$(error $(CONFIG_FILE) not found.)
endif
include $(CONFIG_FILE)

SDS ?= 0
HW ?= 0
ON_THE_FLY=0
AMAJOR=0
REBUILD=0
DEBUG ?= 0
PROF  ?= 0

PLATFORM ?= zed
HARDWARE ?= hw
TARGET_OS?=linux
ifeq ($(SDS), 0)
PLATFORM=cpu
endif
ifeq ($(HW), 0)
HARDWARE=sw
endif
ifeq ($(SDS), 1)
DIR_SUFFIX=$(ELF)-$(HARDWARE)
endif

ROOTDIR=/home/vzhao/SoCaffe/accel
SRCDIR=$(ROOTDIR)/src/
GEMM_SRCDIR=$(SRCDIR)gemm/
GEMV_SRCDIR=$(SRCDIR)gemv/
TEST_SRCDIR=$(SRCDIR)test/
OBJDIR=./obj$(DIR_SUFFIX)/
GEMM_OBJDIR=$(OBJDIR)gemm/
GEMV_OBJDIR=$(OBJDIR)gemv/
TEST_OBJDIR=$(OBJDIR)test/
BINDIR=./bin$(DIR_SUFFIX)/
GEMM_BINDIR=$(BINDIR)gemm/
GEMV_BINDIR=$(BINDIR)gemv/
TEST_BINDIR=$(BINDIR)test/
INCLUDE=$(ROOTDIR)/include/
LIBDIR=./lib/

ACCEL=accel
ACCEL_SHORT=$(ACCEL)-$(PLATFORM)-$(TARGET_OS)-$(HARDWARE)
LIBACCEL_SHORT=lib$(ACCEL_SHORT).so
LIBACCEL=$(LIBACCEL_SHORT)
STATIC_LIBACCEL=lib$(ACCEL_SHORT).a

CC=gcc
CXX=g++
LD=ld
HARDWARE_CC=gcc
HARDWARE_CXX=g++
RELEASE_CC=gcc
RELEASE_CXX=g++
TOOLCHAIN_CC=gcc
TOOLCHAIN_CXX=g++
HARDWARE_CFLAGS=-Wall -Werror -Wno-unknown-pragmas -Wno-unused-label -fPIC -O3
CFLAGS=-fPIC -Ofast -Wall -Werror -Wno-unknown-pragmas -Wno-unused-label
LFLAGS=-L$(LIBDIR) -lm -lpthread -l$(ACCEL_SHORT)

# Configuration
GEMV_BLK_N ?= 32
GEMV_BLK_M ?= 32
CFLAGS += -DGEMV_BLK_N=$(GEMV_BLK_N) -DGEMV_BLK_M=$(GEMV_BLK_M)
SLOW_SYS_PORT ?= 0
ZERO_COPY ?= 0
ifeq ($(SLOW_SYS_PORT), 1)
CFLAGS += -DSLOW_SYS_PORT
endif
ifeq ($(ZERO_COPY), 1)
CFLAGS += -DZERO_COPY
endif
GEMM_SCALE ?= 0
CFLAGS += -DGEMM_SCALE=$(GEMM_SCALE)
GEMM_NUM_PIPE ?= 1
CFLAGS += -DGEMM_NUM_PIPE=$(GEMM_NUM_PIPE)
GEMM_HLS ?= 1
ifeq ($(GEMM_HLS), 1)
CFLAGS += -DGEMM_HLS
endif
GEMM_WITH_ALPHA ?= 1
ifeq ($(GEMM_WITH_ALPHA), 1)
CFLAGS += -DGEMM_WITH_ALPHA
endif
GEMM_NO_ADD_DSP ?= 1
ifeq ($(GEMM_NO_ADD_DSP), 1)
CFLAGS += -DGEMM_NO_ADD_DSP
endif
GEMM_RESOURCE_PARTITION ?= 0
ifeq ($(GEMM_RESOURCE_PARTITION), 1)
CFLAGS += -DGEMM_RESOURCE_PARTITION
endif
GEMM_HALF_FLOAT ?= 0
ifeq ($(GEMM_HALF_FLOAT), 1)
CFLAGS += -DGEMM_HALF_FLOAT
endif
GEMM_RESOURCE_CONSTRAINT ?= 0
ifeq ($(GEMM_RESOURCE_CONSTRAINT), 1)
CFLAGS += -DGEMM_RESOURCE_CONSTRAINT
endif
ifdef GEMM_COPY_METHOD
CFLAGS += -DGEMM_COPY_METHOD=$(GEMM_COPY_METHOD)
endif

TEST_GEMM_SDS=test_gemm_sds
TEST_GEMM_TRANS=test_gemm_trans
TEST_GEMM_PLAIN=test_gemm_plain
TEST_GEMM_BLOCK=test_gemm_block_main
TEST_GEMM_BLOCK_UNIT=test_gemm_block_unit
TEST_GEMV_SDS=test_gemv_sds
ELF=

ifeq ($(DEBUG), 1)
CFLAGS+=-g -ggdb
endif

ifeq ($(PROF), 1)
CFLAGS+=-g -pg
endif

# Macros
CFLAGS+=-I$(INCLUDE)
HARDWARE_CFLAGS+=-I$(INCLUDE)

ifeq ($(SDS), 1)
SDSFLAGS+=-sds-pf $(PLATFORM) 
TOOLCHAIN=arm-xilinx-linux-gnueabi
# CC=$(TOOLCHAIN)-gcc
TOOLCHAIN_CC=$(TOOLCHAIN)-gcc
TOOLCHAIN_CXX=$(TOOLCHAIN)-g++
CC=sdscc $(SDSFLAGS)
CXX=sds++ $(SDSFLAGS)
LD=$(TOOLCHAIN)-ld
HARDWARE_CC=$(CC)
HARDWARE_CXX=$(CXX)
RELEASE_CC=$(CC)
RELEASE_CXX=$(CXX)

ifeq ($(HW), 1)

GEMV_CLOCK_ID := 1
DM_CLOCK_ID := 1
GEMM_CLOCK_ID ?= 1

GEMM_FULL_MODE ?= 0
WITH_GEMV ?= 0
ifeq ($(WITH_GEMV), 1)
SDSFLAGS += -sds-hw gemv_accel $(GEMV_SRCDIR)gemv_accel.cc -clkid $(GEMV_CLOCK_ID) -sds-end
endif
ifeq ($(GEMM_FULL_MODE), 1)
CFLAGS+=-DGEMM_FULL_MODE
SDSFLAGS+= -sds-hw gemm_accel gemm/gemm_accel.c\
					 -files $(GEMM_SRCDIR)gemm_block_unit.c,$(GEMM_SRCDIR)gemm_accel_hls.cc\
					 -clkid $(GEMM_CLOCK_ID) -sds-end
else
	ifeq ($(GEMM_HLS), 1)
		SDSFLAGS+=\
			-sds-hw gemm_accel_full $(GEMM_SRCDIR)gemm_accel_hls.cc -clkid $(GEMM_CLOCK_ID) -sds-end
	else
		SDSFLAGS+=\
			-sds-hw gemm_block_units_mmult $(GEMM_SRCDIR)gemm_block_unit.c -clkid $(GEMM_CLOCK_ID) -sds-end\
			-sds-hw gemm_block_units_mplus $(GEMM_SRCDIR)gemm_block_unit.c -clkid $(GEMM_CLOCK_ID) -sds-end
	endif
endif
SDSFLAGS+=-dmclkid $(DM_CLOCK_ID)
ifeq ($(VERBOSE), 1)
SDSFLAGS+=-verbose
endif

ifeq ($(ESTI), 1)
SDSFLAGS += -perf-funcs gemv_accel
endif
endif

ELF=.elf
endif

GEMM_OBJ=gemm_cpu.o gemm_utils.o gemm_trans.o gemm_plain.o gemm_sds.o gemm_accel.o gemm_accel_hls.o
ifeq ($(GEMM_HLS), 0)
GEMM_OBJ+= gemm_block_unit.o
endif
GEMV_OBJ=gemv_sds.o gemv_accel.o gemv_accel_call.o
ifeq ($(BLOCK), 1)
GEMM_OBJ+=gemm_block.o
endif

GEMM_OBJS=$(addprefix $(GEMM_OBJDIR), $(GEMM_OBJ))
GEMV_OBJS=$(addprefix $(GEMV_OBJDIR), $(GEMV_OBJ))
DEPS=$(wildcard $(INCLUDE)/*.h)\
		 $(wildcard $(INCLUDE)/*.hh)

all: obj bin lib  

lib: $(LIBDIR)$(LIBACCEL)

$(LIBDIR)$(LIBACCEL): $(GEMM_OBJS) $(GEMV_OBJS)
	@ echo "RELEASE_CXX -o $@"
	@ $(RELEASE_CXX) $(CFLAGS) $^ -shared -o $@

$(GEMM_OBJDIR)%.o: $(GEMM_SRCDIR)%.c $(DEPS)
	@ echo "CXX -o $@"
	@ $(CXX) $(CFLAGS) -c $< -o $@

$(GEMM_OBJDIR)%.o: $(GEMM_SRCDIR)%.cc $(DEPS)
	@ echo "CXX -o $@"
	@ $(CXX) $(CFLAGS) -c $< -o $@

$(GEMV_OBJDIR)%.o: $(GEMV_SRCDIR)%.cc $(DEPS)
	@ echo "CXX -o $@"
	@ $(CXX) $(CFLAGS) -c $< -o $@

TESTS=test_unit test_sds test_trans test_plain test_gemv_sds
ifeq ($(BLOCK), 1)
TESTS+=test_block
endif

# TEST builds
test: obj bin lib $(TESTS)

test_trans: obj bin $(TEST_BINDIR)$(TEST_GEMM_TRANS)$(ELF)
test_sds: 	obj bin $(TEST_BINDIR)$(TEST_GEMM_SDS)$(ELF)
test_unit: 	obj bin $(TEST_BINDIR)$(TEST_GEMM_BLOCK_UNIT)$(ELF)
test_block: obj bin $(TEST_BINDIR)$(TEST_GEMM_BLOCK)$(ELF)
test_plain: obj bin $(TEST_BINDIR)$(TEST_GEMM_PLAIN)$(ELF)
test_gemv_sds:	obj bin $(TEST_BINDIR)$(TEST_GEMV_SDS)$(ELF)

$(TEST_BINDIR)$(TEST_GEMM_TRANS)$(ELF): $(TEST_OBJDIR)$(TEST_GEMM_TRANS).o 
	@ echo "TOOLCHAIN_CXX -o $@"
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_BLOCK)$(ELF): $(TEST_OBJDIR)$(TEST_GEMM_BLOCK).o
	@ echo "TOOLCHAIN_CXX -o $@"
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_PLAIN)$(ELF): $(TEST_OBJDIR)$(TEST_GEMM_PLAIN).o
	@ echo "TOOLCHAIN_CXX -o $@"
	@ # $(RELEASE_CXX) $(CFLAGS) $^ -o $@
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_BLOCK_UNIT)$(ELF): $(TEST_OBJDIR)$(TEST_GEMM_BLOCK_UNIT).o
	@ echo "TOOLCHAIN_CXX -o $@"
	@ # $(RELEASE_CXX) $(CFLAGS) $^ -o $@
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_SDS)$(ELF): $(TEST_OBJDIR)$(TEST_GEMM_SDS).o
	@ echo "TOOLCHAIN_CXX -o $@"
	@ # $(RELEASE_CXX) $(CFLAGS) $^ -o $@
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMV_SDS)$(ELF): $(TEST_OBJDIR)$(TEST_GEMV_SDS).o
	@ echo "TOOLCHAIN_CXX -o $@"
	@ # $(RELEASE_CXX) $(CFLAGS) $^ -o $@
	@ $(TOOLCHAIN_CXX) $(CFLAGS) $< -o $@ $(LFLAGS)

$(TEST_OBJDIR)%.o: $(TEST_SRCDIR)%.c $(DEPS)
	@ echo "TOOLCHAIN_CXX -o $@"
	@ $(CXX) $(CFLAGS) -c $< -o $@

$(TEST_OBJDIR)%.o: $(TEST_SRCDIR)%.cc $(DEPS)
	@ echo "TOOLCHAIN_CXX -o $@"
	@ $(CXX) $(CFLAGS) -c $< -o $@

.PHONY: obj bin tar clean

obj:
	@ mkdir -p $(LIBDIR)
	@ mkdir -p $(OBJDIR)
	@ mkdir -p $(GEMM_OBJDIR)
	@ mkdir -p $(TEST_OBJDIR)
	@ mkdir -p $(GEMV_OBJDIR)

bin:
	@ mkdir -p $(BINDIR)
	@ mkdir -p $(GEMM_BINDIR)
	@ mkdir -p $(TEST_BINDIR)
	@ mkdir -p $(GEMV_OBJDIR)

clean:
	@ rm -rf $(OBJS) $(EXEC) $(TEST_GEMM_GRID)$(ELF)
	@ rm -rf _sds *.bit *.elf* sd_card/
	@ rm -rf $(OBJDIR) 
	@ rm -rf $(BINDIR)
	@ rm -rf $(LIBDIR)
