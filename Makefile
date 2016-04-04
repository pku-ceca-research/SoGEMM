VERSION=v0.1.0
DATE=$(shell date +'%y%m%d_%H%M%S')

SDS=0
HW=0
ON_THE_FLY=0

CC=gcc
CFLAGS=-Ofast
LFLAGS=-lm

VPATH=./src/
OBJDIR=./obj/
EXEC=gemm
TEST_GEMM_GRID=test_gemm_grid
ELF=

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
	-sds-hw gemm_mmult gemm_grid.c -sds-end\
	-sds-hw gemm_madd  gemm_grid.c -sds-end
ifeq ($(USE_PIPELINE), 1)
CFLAGS+=-DUSE_PIPELINE
endif

# SDSFLAGS+=gemm_sds gemm_onsds gemm_nn_sds gemm_nt_sds gemm_tn_sds gemm_tt_sds 
# SDSFLAGS+=-sds-end 
endif

CC=sdscc $(SDSFLAGS)
EXEC=gemm.elf
ELF=.elf
endif

OBJ=gemm.o gemm_grid.o gemm_utils.o gemm_trans.o

OBJS=$(addprefix $(OBJDIR), $(OBJ))
DEPS=$(wildcard src/*.h) $(wildcard src/*.c) Makefile

all: obj $(EXEC) 

test: obj $(TEST_GEMM_GRID)$(ELF)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ src/main.c -o $@ $(LFLAGS)

$(TEST_GEMM_GRID)$(ELF): $(OBJS)
	$(CC) $(CFLAGS) $^ src/$(TEST_GEMM_GRID).c -o $@ $(LFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

tar:
	tar cvf GEMM_$(VERSION)_$(DATE).tar.gz ../GEMM

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) $(TEST_GEMM_GRID)$(ELF)
	rm -rf _sds *.bit *.elf sd_card/
	rm -rf obj
