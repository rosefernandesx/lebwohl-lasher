"""
Basic Python Lebwohl-Lasher code accelerated using cython.
No changes were made to the python code, but setup and run 
scripts used to analyse speed up provided from running in C 
only.  
Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python setup.py build_ext -fi
python run.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

RF 27-10-2026
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cimport cython
cimport openmp
from libc.math cimport cos, exp, sin
from cython.parallel cimport prange

#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double one_energy(
    double[:, ::1] arr,
    Py_ssize_t ix,
    Py_ssize_t iy,
    Py_ssize_t nmax
) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    cdef double en = 0.0
    cdef double ang
    cdef double c
    cdef Py_ssize_t ixp, ixm, iyp, iym

    # Periodic boundary conditions
    ixp = 0 if ix == nmax-1 else ix+1
    ixm = nmax-1 if ix==0 else ix-1
    iyp = 0 if iy == nmax-1 else iy+1
    iym = nmax-1 if iy == 0 else iy-1
#
# Add together the 4 neighbour contributions
# to the energy
#
    # Updated to use cos from libc.math instead of numpy
    # Also avoids using square
    ang = arr[ix,iy]-arr[ixp,iy]
    c = cos(ang)
    en += 0.5*(1.0 - 3.0*c*c)

    ang = arr[ix,iy]-arr[ixm,iy]
    c = cos(ang)
    en += 0.5*(1.0 - 3.0*c*c)

    ang = arr[ix,iy]-arr[ix,iyp]
    c = cos(ang)
    en += 0.5*(1.0 - 3.0*c*c)

    ang = arr[ix,iy]-arr[ix,iym]
    c = cos(ang)
    en += 0.5*(1.0 - 3.0*c*c)

    return en
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def all_energy(
    double[:, ::1] arr,
    Py_ssize_t nmax
):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef double enall = 0.0
    cdef Py_ssize_t i, j
    with nogil:
        for i in prange(nmax, schedule='static'):
            for j in range(nmax):
                enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def get_order(
    double[:, ::1] arr,
    Py_ssize_t nmax
):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    arr_np = np.asarray(arr)
    lab_np = np.ascontiguousarray(
        np.vstack((np.cos(arr_np),np.sin(arr_np),np.zeros_like(arr_np))).reshape(3,nmax,nmax),
        dtype=np.float64)
    cdef double [:, :, ::1] lab = lab_np
    cdef Py_ssize_t a, b, i, j
    cdef double acc
    for a in range(3):
        for b in range(a,3):
            acc = 0.0
            with nogil:
                for i in prange(nmax, schedule='static'):
                    for j in range(nmax):
                        acc += 3.0*lab[a,i,j]*lab[b,i,j]
            Qab[a,b] = acc - delta[a,b]*nmax*nmax
            Qab[b,a] = Qab[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.real.max()
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def MC_step(
    double [:, ::1] arr,
    double Ts,
    Py_ssize_t nmax
):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    scale=0.1+Ts
    cdef Py_ssize_t accept = 0
    aran_np = np.random.normal(scale=scale, size=(nmax,nmax))
    uran_np = np.random.uniform(0.0, 1.0, size=(nmax,nmax))
    cdef double[:, ::1] aran = aran_np
    cdef double[:, ::1] uran = uran_np
    cdef Py_ssize_t i, j, parity
    cdef double ang, en0, en1, boltz
    with nogil:
        for parity in range(2):
            for i in prange(nmax, schedule='static'):
                for j in range((i+parity) & 1, nmax, 2):
                    ang = aran[i,j]
                    en0 = one_energy(arr,i,j,nmax)
                    arr[i,j] += ang
                    en1 = one_energy(arr,i,j,nmax)
                    if en1<=en0:
                        accept += 1
                    else:
                    # Now apply the Monte Carlo test - compare
                    # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                        boltz = exp( -(en1 - en0) / Ts )

                        if boltz >= uran[i,j]:
                            accept += 1
                        else:
                            arr[i,j] -= ang
    return accept/(<double>(nmax*nmax))
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def main(program, nsteps, nmax, temp, pflag, nthreads=0):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    if nmax %2 != 0:
        raise ValueError("SIZE must be even for the checkerboard OpenMP update.")
    if nthreads > 0:
        openmp.omp_set_num_threads(nthreads)
    nthreads_used = openmp.omp_get_max_threads()

    # Create and initialise lattice
    lattice = np.ascontiguousarray(initdat(nmax), dtype=np.float64)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1, dtype=float)
    ratio = np.zeros(nsteps+1, dtype=float)
    order = np.zeros(nsteps+1, dtype=float)

    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Threads: {:d}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],nthreads_used,runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
