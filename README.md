# 474-Project-2
Project 2: Calculate each row and column of a Dynamic Time-Warping distance matrix in paralell using MPI

Group members:

Jamie Lambrecht mjlambrecht@csu.fullerton.edu

## Overview
This is a distributed program using MPI (mpi4py) with a manager process and two agent processes. The purpose of this program is to calculate the dynamic time-warping distance. This is an algorithm for finding an optimal alignment between time-series with different frequencies (e.g. two vectors of point coordinates representing similar motion at different speed). This kind of data can be used for classification in machine learning algorithms, such as identifying hand gestures, etc.

## Installation & Usage
To run the program, clone the source repository and run the command "mpiexec -n 1 python3.9 -m mpi4py manager.py" in the root directory of the repository using the Python interpretor. You must install the correct version of python (3.9) as well mpi4py and any of its dependencies (mpich or openmpi, gcc, etc.

## Requirements
This program was built and tested using Python 3.9. The libraries were incompatible with Python 3.10. You must install mpi4py via Pip.

## Current Implementation
Currently, all the sends and receives are non-blocking and there is redundancy when columns and rows overlap. The original idea was to use locks to synchronize starting points between the row agent and the column agent. I still intend to extend the current implementation to do this sometime in the near future. As of now, the row agent tends to complete most of the work before the column agent finishes a column, although, if a call to time.sleep(5) is inserted in the main loop of agent.py on the condition that rank == 1, the column agent will finish, demonstrating that the program is at least working as intended. I am honestly not sure if the row agent being faster is due to the row code preceding the column code, or if the MPI (or perhaps the Linux kernel) scheduler gives higher priority to the first spawned process, or if it is a machine or platform-dependent situation. This repo will be updated if/when I make it so that an agent sends a signal to block the other when it finishes and have the manager give the other agent a shorter starting point, which was the original idea that I had for this program.  

## Original Design Idea
** The following was the intended pattern for the school project. The final product ended up different than the original design idea. **

Each agent has a starting point variable based on the subsequence (row or column) index number that is sent to the other on writing to the first cell in its subset.

There is a critical section when an agent begins a new iteration. It must send a request for its starting point from a queue held by the manager process.

If the manager process receives that request it will send a message to initiate blocking in the other agent's routine so that it does not continue until the first agent has received its starting point and begun its iteration.

