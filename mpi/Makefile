SW =GNU3
VAMP = false
SCOREP = false

# default values
LDFLAGS := -fopenmp -g # -pg

VERBOSE = false
VERTEX_REORDERING = 0
REAL_BENCHMARK = false
ifeq ($(VERBOSE), false)
VERBOSE_OPT = -DVERVOSE_MODE=0
else
VERBOSE_OPT = -DVERVOSE_MODE=1
endif
VERTEX_REORDERING_OPT = -DVERTEX_REORDERING=$(VERTEX_REORDERING)
ifeq ($(REAL_BENCHMARK), false)
REAL_BENCHMARK_OPT =
else
REAL_BENCHMARK_OPT = -DREAL_BENCHMARK
endif


GCC_BASE := -fopenmp -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math -msse4.2 $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT) # -pg
#GCC_BASE := -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math -msse4.2 # -pg
FCC_BASE := -Kopenmp -Xg -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -g #-Nquickdbg=heapchk # -Koptmsg=2
#FCC_BASE := -Xg -g -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS

MPICC := mpicc
MPICPP := mpicxx

BINS := runnable
OBJS := main.o low_level_func.o ../generator/splittable_mrg.o
LIBS := -lm

ifeq ($(SW), GNU3)
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), GNU0)
CFLAGS := $(GCC_BASE) -O0
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), FUJI3)
ifeq ($(PJM_JOBNAME), STDIN)
MPICC := mpifcc
MPICPP := mpiFCC
else
MPICC := mpifccpx
MPICPP := mpiFCCpx
endif
CFLAGS := $(FCC_BASE) -Kfast -Klargepage -O3 -DNDEBUG
LDFLAGS := $(CFLAGS)
endif
ifeq ($(SW), FUJI0)
ifeq ($(PJM_JOBNAME), STDIN)
MPICC := mpifcc
MPICPP := mpiFCC
else
MPICC := mpifccpx
MPICPP := mpiFCCpx
endif
CFLAGS := $(FCC_BASE) -g -O0
LDFLAGS := $(CFLAGS)
endif

ifeq ($(SW), UVGNU3)
MPICC := mpicc-gcc4.7
MPICPP := mpicxx-gcc4.7
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG
LIBS := $(LIBS) -lnuma -lmpi
endif
ifeq ($(SW), UVGNU0)
MPICC := mpicc-gcc4.7
MPICPP := mpicxx-gcc4.7
CFLAGS := $(GCC_BASE) -O0
LIBS :=  $(LIBS) -lnuma -lmpi
endif
ifeq ($(SW), INTEL3)
MPICC := icc
MPICPP := icpc
CFLAGS := -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG
#CFLAGS := -ipo -openmp -g3 -Drestrict=__restrict__ -O2 -DNDEBUG
LDFLAGS := -openmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif
ifeq ($(SW), INTEL0)
MPICC := icc
MPICPP := icpc
CFLAGS := -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O0
LDFLAGS := -openmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif
ifeq ($(SW), FXGNU3)
MPICC := gcc
MPICPP := g++
CFLAGS := -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math # -g -pg
LDFLAGS := -fopenmp -g # -g -pg
INC := $(INC) -I/opt/FJSVtclang/GM-1.2.0-16/include/mpi/fujitsu
LIBS := -L/opt/FJSVtclang/GM-1.2.0-16/lib64 -lm -lmpi -lmpi_cxx #-ltrt -ltrtmetcpp -ltrtmet_c
endif
ifeq ($(SW), FXGNU0)
MPICC := gcc
MPICPP := g++
CFLAGS := -fopenmp -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS := -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
LDFLAGS := -fopenmp -g # -g -pg
INC := $(INC) -I/opt/FJSVtclang/GM-1.2.0-16/include/mpi/fujitsu
LIBS := -L/opt/FJSVtclang/GM-1.2.0-16/lib64 -lm -lmpi -lmpi_cxx #-ltrt -ltrtmetcpp -ltrtmet_c
endif

ifeq ($(VAMP), true)
MPICC := vtcc -vt:cc $(MPICC) -vt:inst manual -vt:hyb -DVTRACE
MPICPP := vtcc -vt:cc $(MPICPP) -vt:inst manual -vt:hyb -DVTRACE
endif
ifeq ($(SCOREP), true)
MPICC := scorep --nocompiler --user --static $(MPICC) -DSCOREP
MPICPP := scorep --nocompiler --user --static $(MPICPP) -DSCOREP
endif

CPPFLAGS := $(CFLAGS)

all: cpu

cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)

gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o
#	g++ -c -g -O3 -Wall $< -o $*.o

.SUFFIXES: .o .c .cpp .cc

.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o

.cpp.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)
