"""
Run the compiled Lebwohl-Lasher simulation.

Usage: python run_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
"""
import sys
from CythonLL import main

if len(sys.argv) != 5:
    raise SystemExit(
        f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>"
    )

program = sys.argv[0]
iterations = int(sys.argv[1])
size = int(sys.argv[2])
temperature = float(sys.argv[3])
plotflag = int(sys.argv[4])
main(program, iterations, size, temperature, plotflag)
