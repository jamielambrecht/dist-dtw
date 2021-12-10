# 474-Project-2
Project 2: Calculate each row and column of a Dynamic Time-Warping distance matrix in paralell using MPI

Group members:

Jamie Lambrecht mjlambrecht@csu.fullerton.edu

## Overview
This is a distributed program with a manager process and two agent processes. The purpose of this program is to calculate the dynamic time-warping distance. This is an algorithm for finding an optimal alignment between time-series with different frequencies (e.g. two vectors of point coordinates representing similar motion at different speed). This kind of data can be used for classification in machine learning algorithms, such as identifying hand gestures, etc.

Each agent has a starting point variable based on the subsequence (row or column) index number that is sent to the other on writing to the first cell in its subset.
There is a critical section when an agent begins a new iteration. It must send a request for its starting point from a queue held by the manager process.

If the manager process receives that request it will send a message to initiate blocking in the other agents routine so that it does not continue until the first agent has received its starting point and begun its iteration.

To run the program, clone the source repository and run manager.py using the Python interpretor. You must install the correct version of python (3.9) as well mpi4py and any of its dependencies (mpich or openmpi, gcc, etc).

## Requirements
This program was built and tested using Python 3.9. The libraries were incompatible with Python 3.10. You must install mpi4py via Pip.

## Usage
Run "python manager.py".