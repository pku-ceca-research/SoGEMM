VERSION=v0.1.0
DATE=$(shell date +'%y%m%d_%H%M%S')
NAME=

SDS=0
HW=0
ON_THE_FLY=0
AMAJOR=0
REBUILD=0
DEBUG=0
PROF=0

CC=gcc
CFLAGS=-Ofast -Wall -Werror -Wno-unknown-pragmas
LFLAGS=-lm

VPATH=./src/
SRCDIR=./src/
OBJDIR=./obj$(ELF)/
INCLUDE=./include/
EXEC=gemm
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

BUFTYPE=1
# Macros
CFLAGS+=-DBUFTYPE=$(BUFTYPE)
CFLAGS+=-I$(INCLUDE)

ifeq ($(AMAJOR), 1)
	CFLAGS+=-DAMAJOR
endif
# on the fly blocking version
ifeq ($(ON_THE_FLY), 1)
CFLAGS+=-DBLK_ON_THE_FLY
endif

ifeq ($(SDS), 1)
CFLAGS+=-DSDS

PLATFORM=zed
SDSFLAGS=-sds-pf $(PLATFORM) 

ifeq ($(HW), 1)
SDSFLAGS+=\
	-sds-hw gemm_block_units_mmult gemm_block_unit.c -files src/gemm_block.c -clkid 2 -sds-end\
	-sds-hw gemm_block_units_mplus gemm_block_unit.c -files src/gemm_block.c -clkid 2 -sds-end
ifeq ($(USE_PIPELINE), 1)
CFLAGS+=-DUSE_PIPELINE
endif

ifeq ($(REBUILD), 1)
SDSFLAGS+=-rebuild-hardware
endif

endif

CC=sdscc $(SDSFLAGS)
EXEC=gemm.elf
ELF=.elf
endif

OBJ=gemm_cpu.o gemm_utils.o gemm_trans.o\
		gemm_block.o gemm_plain.o\
		gemm_block_unit.o gemm_sds.o
OBJS=$(addprefix $(OBJDIR), $(OBJ))
DEPS=$(wildcard $(INCLUDE)/*.h) $(wildcard $(SRCDIR)/*.c) Makefile

all: obj test 

test: obj test_unit test_block test_sds test_trans test_plain

test_trans: obj $(TEST_GEMM_TRANS)$(ELF)
test_sds: 	obj $(TEST_GEMM_SDS)$(ELF)
test_unit: 	obj $(TEST_GEMM_BLOCK_UNIT)$(ELF)
test_block: obj $(TEST_GEMM_BLOCK)$(ELF)
test_plain: obj $(TEST_GEMM_PLAIN)$(ELF)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ src/main.c -o $@ $(LFLAGS)

$(TEST_GEMM_TRANS)$(ELF): $(OBJS) $(OBJDIR)$(TEST_GEMM_TRANS).o
	$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_GEMM_BLOCK)$(ELF): $(OBJS) $(OBJDIR)$(TEST_GEMM_BLOCK).o
	$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_GEMM_PLAIN)$(ELF): $(OBJS) $(OBJDIR)$(TEST_GEMM_PLAIN).o
	$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_GEMM_BLOCK_UNIT)$(ELF): $(OBJS) $(OBJDIR)$(TEST_GEMM_BLOCK_UNIT).o
	$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(TEST_GEMM_SDS)$(ELF): $(OBJS) $(OBJDIR)$(TEST_GEMM_SDS).o
	$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p $(OBJDIR)

tar:
	tar cvf GEMM_$(VERSION)_$(DATE)_$(NAME).tar.gz ../GEMM/sd_card ../GEMM/_sds/reports ../GEMM/src

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) $(TEST_GEMM_GRID)$(ELF)
	rm -rf _sds *.bit *.elf sd_card/
	rm -rf $(OBJDIR) 
	rm -rf test_*
