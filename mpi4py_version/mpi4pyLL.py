"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python mpi4pyLL.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

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

SH 16-Oct-23

The code has been updated to port to mpi4py and run on multiple cores.

Run with:
    mpiexec -n <NPROCS> python mpi4pyLL.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

RF 5-11-26
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI

# MPI set up 
comm =  MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()
numworkers = numtasks - 1

MASTER = 0
BEGIN = 1
LTAG = 2
RTAG = 3
DONE = 4
STATS = 5
NSTATS = 11

#=======================================================================
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
    arr=np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
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
def one_energy(arr,ix,iy,nmax):
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
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def exchange_halos(arr, nmax, offset, rows, above, below):
    if above==taskid:
        return
    top = offset
    bottom = offset+rows-1
    haloup = (offset-1)%nmax
    halodown = (offset+rows)%nmax
    req1 = comm.Isend([arr[top,:], nmax,MPI.DOUBLE], dest=above, tag=RTAG)
    req2 = comm.Isend([arr[bottom,:],nmax,MPI.DOUBLE], dest=below, tag=LTAG)
    comm.Recv([arr[haloup,:],nmax,MPI.DOUBLE], source=above, tag=LTAG)
    comm.Recv([arr[halodown,:],nmax,MPI.DOUBLE], source=below, tag=RTAG)
    MPI.Request.Waitall([req1,req2])
#=======================================================================
def all_energy(arr,nmax,offset,rows,above,below):
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
    exchange_halos(arr,nmax,offset,rows,above,below)
    enall = 0.0
    for i in range(offset,offset+rows):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
def get_order(arr,nmax,offset,rows):
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
    mine = arr[offset:offset+rows,:]
    lab = np.vstack((np.cos(mine),np.sin(mine),np.zeros_like(mine))).reshape(3,rows,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(rows):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    return Qab
#=======================================================================
def MC_step(arr,Ts,nmax,offset,rows,above,below,rng):
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
    accept = 0
    nhalf = nmax//2
    for parity in range(2):
        exchange_halos|(arr,nmax,offset,rows,above,below)
        xran = rng.intergers(offset,high=offset+rows,size=(rows,nhalf))
        yran = rng.integeres(0,high=nhalf,size=(rows,nhalf))
        aran = rng.normal(scale=scale, size = (rows,nhalf))
        for i in range(rows):
            for j in range(nhalf):
                ix = xran[i,j]
                iy = 2*yran[i,j] + (parity-ix)%2
                ang = aran[i,j]
                en0 = one_energy(arr,ix,iy,nmax)
                arr[ix,iy] += ang
                en1 = one_energy(arr,ix,iy,nmax)
                if en1<=en0:
                    accept += 1
                else:
                # Now apply the Monte Carlo test - compare
                # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                    boltz = np.exp( -(en1 - en0) / Ts )

                    if boltz >= np.random.uniform(0.0,1.0):
                        accept += 1
                    else:
                        arr[ix,iy] -= ang
    return accept
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
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
    if numworkers<1 or nmax%2!=0 or nmax<numworkers:
        if taskid==MASTER:
            print(f"Need a minimum of two tasks, an even size and size >= number of workers: tasks={numtasks}, SIZE={max}")
            sys.exit(1)

    lattice = np.zeros((nmax,nmax))

    if taskid==MASTER:
        # Create and initialise lattice
        lattice = initdat(nmax)
        # Plot initial frame of lattice
        plotdat(lattice,pflag,nmax)
        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1, dtype=float)
        ratio = np.zeros(nsteps+1, dtype=float)
        order = np.zeros(nsteps+1, dtype=float)
        # Calculate rows for workers and send them out
        averow = nmax//numworkers
        extra = nmax%numworkers
        seed = np.ranodm.SeedSequence().entropy
        offset = 0
        for i in range(1,numworkers+1):
            rows=averow
            if i<extra:
                rows+=1
            above = i-1 if i>1 else numworkers
            below = i+1 if i<numworkers else 1
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)
            comm.send(seed, dest=i, tag=BEGIN)
            comm.send([lattice[offset:offset+rows,:],rows*nmax,MPI.DOUBLE],dest=i, tag=BEGIN)
            offset += rows

        # Begin doing and timing some MC steps.
        stats=np.zeros(NSTATS)
        comm.Barrier
        initial = time.time()
        for it in range(1,nsteps+1):
            tot = np.zeros(NSTATS)
            for i in range(1,numworkers+1):
                comm.Recv([stats,NSTATS,MPI.DOUBLE], source = i, tag = STATS)
                tot += stats
            energy[it] = tot[1]
            if it==0:
                ratio[0] = 0.5 # ideal value
            else:
                ratio[it] = tot[0]/(nmax*nmax)
            Qab=tot[2:].reshape(3,3)/(2*nmax*nmax)
            eigenvalues,eigenvectors = np.linalg.eig(Qab)        
            order[it] = eigenvalues.max()
        final = time.time()
        runtime = final-initial

        # wait for final rows from all workers
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([lattice[offset:offset+rows,:],rows*nmax,MPI.DOUBLE], source=i, tag=DONE)

        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)

    else: # Worker only
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)
        seed = comm.recv(source=MASTER, tag=BEGIN)
        comm.Recv([lattice[offset:offset+rows,:],rows*nmax,MPI.DOUBLE], source=MASTER, tag=BEGIN)
        rng = np.random.default_rng([seed,taskid])   # independent stream per worker

        stats = np.zeros(NSTATS)
        comm.Barrier()
        for it in range(nsteps+1):
            accept = 0
            if it>0:
                accept = MC_step(lattice,temp,nmax,offset,rows,above,below,rng)
            stats[0] = accept
            stats[1] = all_energy(lattice,nmax,offset,rows,above,below)
            stats[2:] = get_order(lattice,nmax,offset,rows).ravel()
            comm.Send([stats,NSTATS,MPI.DOUBLE], dest=MASTER, tag=STATS)

        # Finally, send this worker's rows of the final lattice back to the master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([lattice[offset:offset+rows,:],rows*nmax,MPI.DOUBLE], dest=MASTER, tag=DONE)
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
    elif taskif==MASTER:
        print("Usage: mpiexec -n <NTASKS> python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
