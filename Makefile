VERSION=1.0.0
DATE=$(shell date +'%y%m%d_%H%M%S')
NAME=

SDS=0
HW=0
ON_THE_FLY=0
AMAJOR=0
REBUILD=0
DEBUG=0
PROF=0

PLATFORM=zed
HARDWARE=hw
TARGET_OS=linux
ifeq ($(SDS), 0)
PLATFORM=cpu
endif
ifeq ($(HW), 0)
HARDWARE=sw
endif
ifeq ($(SDS), 1)
DIR_SUFFIX=$(ELF)-$(HARDWARE)
endif

SRCDIR=./src/
GEMM_SRCDIR=$(SRCDIR)gemm/
TEST_SRCDIR=$(SRCDIR)test/
OBJDIR=./obj$(DIR_SUFFIX)/
GEMM_OBJDIR=$(OBJDIR)gemm/
TEST_OBJDIR=$(OBJDIR)test/
BINDIR=./bin$(DIR_SUFFIX)/
GEMM_BINDIR=$(BINDIR)gemm/
TEST_BINDIR=$(BINDIR)test/
INCLUDE=./include/
LIBDIR=./lib/

ACCEL=accel
ACCEL_SHORT=$(ACCEL)-$(PLATFORM)-$(TARGET_OS)-$(HARDWARE)
LIBACCEL_SHORT=lib$(ACCEL_SHORT).so
LIBACCEL=$(LIBACCEL_SHORT).$(VERSION)
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
HARDWARE_CFLAGS=-Wall -Werror -Wno-unknown-pragmas -fPIC -O3
CFLAGS=-fPIC -Ofast -Wall -Werror -Wno-unknown-pragmas
LFLAGS=-L$(LIBDIR) -lm -lpthread -l$(ACCEL_SHORT)

TEST_GEMM_SDS=test_gemm_sds
TEST_GEMM_TRANS=test_gemm_trans
TEST_GEMM_PLAIN=test_gemm_plain
TEST_GEMM_BLOCK=test_gemm_block_main
TEST_GEMM_BLOCK_UNIT=test_gemm_block_unit
ELF=

ifeq ($(DEBUG), 1)
CFLAGS=-static -g -ggdb
endif

ifeq ($(PROF), 1)
CFLAGS=-g -pg
endif

# Macros
CFLAGS+=-I$(INCLUDE)
HARDWARE_CFLAGS+=-I$(INCLUDE)

# ifeq ($(AMAJOR), 1)
# 	CFLAGS+=-DAMAJOR
# endif

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
SDSFLAGS+=\
	-sds-hw gemm_block_units_mmult gemm/gemm_block_unit.c -clkid 2 -sds-end\
	-sds-hw gemm_block_units_mplus gemm/gemm_block_unit.c -clkid 2 -sds-end\
	-dmclkid 2
endif

ELF=.elf
endif

GEMM_OBJ=gemm_cpu.o gemm_utils.o gemm_trans.o gemm_block.o gemm_plain.o gemm_sds.o gemm_block_unit.o

GEMM_OBJS=$(addprefix $(GEMM_OBJDIR), $(GEMM_OBJ))
DEPS=$(wildcard $(INCLUDE)/*.h)

all: obj bin lib  

lib: $(LIBDIR)$(LIBACCEL)

$(LIBDIR)$(LIBACCEL): $(GEMM_OBJS)
	$(RELEASE_CXX) $(CFLAGS) $^ -shared -o $@
	ln -sf $(LIBACCEL) $(LIBDIR)$(LIBACCEL_SHORT)

$(GEMM_OBJDIR)%.o: $(GEMM_SRCDIR)%.c $(DEPS)
	$(CXX) $(CFLAGS) -c $< -o $@

# TEST builds
test: obj bin lib test_unit test_block test_sds test_trans test_plain

test_trans: obj bin $(TEST_BINDIR)$(TEST_GEMM_TRANS)$(ELF)
test_sds: 	obj bin $(TEST_BINDIR)$(TEST_GEMM_SDS)$(ELF)
test_unit: 	obj bin $(TEST_BINDIR)$(TEST_GEMM_BLOCK_UNIT)$(ELF)
test_block: obj bin $(TEST_BINDIR)$(TEST_GEMM_BLOCK)$(ELF)
test_plain: obj bin $(TEST_BINDIR)$(TEST_GEMM_PLAIN)$(ELF)

$(TEST_BINDIR)$(TEST_GEMM_TRANS)$(ELF): $(GEMM_OBJS) $(TEST_OBJDIR)$(TEST_GEMM_TRANS).o
	$(TOOLCHAIN_CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_BLOCK)$(ELF): $(GEMM_OBJS) $(TEST_OBJDIR)$(TEST_GEMM_BLOCK).o
	$(TOOLCHAIN_CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_PLAIN)$(ELF): $(GEMM_OBJS) $(TEST_OBJDIR)$(TEST_GEMM_PLAIN).o
	$(TOOLCHAIN_CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_BLOCK_UNIT)$(ELF): $(GEMM_OBJS) $(TEST_OBJDIR)$(TEST_GEMM_BLOCK_UNIT).o
	$(TOOLCHAIN_CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_BINDIR)$(TEST_GEMM_SDS)$(ELF): $(GEMM_OBJS) $(TEST_OBJDIR)$(TEST_GEMM_SDS).o
	$(TOOLCHAIN_CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_OBJDIR)%.o: $(TEST_SRCDIR)%.c $(DEPS)
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY: obj bin tar clean

obj:
	@ mkdir -p $(LIBDIR)
	@ mkdir -p $(OBJDIR)
	@ mkdir -p $(GEMM_OBJDIR)
	@ mkdir -p $(TEST_OBJDIR)

bin:
	@ mkdir -p $(BINDIR)
	@ mkdir -p $(GEMM_BINDIR)
	@ mkdir -p $(TEST_BINDIR)

tar:
	tar cvf GEMM_$(VERSION)_$(DATE)_$(NAME).tar.gz ../GEMM/sd_card ../GEMM/_sds/reports ../GEMM/src

clean:
	@ rm -rf $(OBJS) $(EXEC) $(TEST_GEMM_GRID)$(ELF)
	@ rm -rf _sds *.bit *.elf sd_card/
	@ rm -rf $(OBJDIR) 
	@ rm -rf $(BINDIR)
	@ rm -rf $(LIBDIR)
