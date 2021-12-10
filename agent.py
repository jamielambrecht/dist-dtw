from mpi4py import MPI
import numpy as np
from numpy.lib.shape_base import column_stack

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

if size != 2:
    print("This program requires exactly 3 processes, one manager and two agents. " + "I am agent {} / {}".format(rank + 1, size))

print(rank)

dimensions = np.array((0,0))
comm.Bcast(dimensions, root=0)

print("{} / {} just received: ".format(rank + 1, size) + str(dimensions))

cost_matrix = np.empty(dimensions, np.int64)
comm.Bcast(cost_matrix, root=0)

print("Just received:\n" + str(cost_matrix))

current_row_index, current_column_index = 0, 0

first_run = True

try:
    if rank == 0:
        while current_row_index < dimensions[1]:
            if first_run:
                previous_row = np.empty(dimensions[1]-1, dtype=np.int64)
            else:
                previous_row = np.empty(dimensions[1]-1 - current_row_index, dtype=np.int64)
            current_row = np.zeros(len(previous_row), dtype=np.int64)
            left_neighbor = np.array(1, dtype=np.int64)
            current_column_index = np.array(1, dtype=np.int64)
            current_row_index = np.array(1, dtype=np.int64)


            reqs = [] 
            reqs.append(comm.Irecv(current_row_index, source=0, tag=0))
            reqs.append(comm.Irecv(current_column_index, source=0, tag=1))
            reqs.append(comm.Irecv(left_neighbor, source=0, tag=2))
            reqs.append(comm.Irecv(previous_row, source=0, tag=3))

            for req in reqs:
                req.wait()

            print("Processing for previous row: " + str(previous_row))
            print("Left neighbor: " + str(left_neighbor))
            print("current_column_index: " + str(current_column_index))
            print("current_row_index: " + str(current_row_index))

            if current_row_index > 0:
                for i in range(0, len(previous_row)):
                    current_row[i] = min(previous_row[i], left_neighbor, previous_row[i - 1]) + cost_matrix[current_column_index, current_row_index]
                    left_neighbor = current_row[i]
            else:
                for i in range(0, len(previous_row)):
                    current_row[i] = left_neighbor + cost_matrix[current_column_index, i + 1]
                    left_neighbor = current_row[i]

            print("Finished processing current row: " + str(current_row))

            comm.Send(current_row, dest=0, tag=8)

            first_run = False
            

    if rank == 1:
        while current_column_index < dimensions[0]:
            if first_run:
                previous_column = np.empty(dimensions[0]-1, dtype=np.int64)
            else:
                previous_column = np.empty(dimensions[0]-1 - current_column_index, dtype=np.int64)
            current_column = np.zeros(len(previous_column), dtype=np.int64)
            up_neighbor = np.array(1, dtype=np.int64)
            current_column_index = np.array(1, dtype=np.int64)
            current_row_index = np.array(1, dtype=np.int64)


            reqs = [] 
            reqs.append(comm.Irecv(current_row_index, source=0, tag=4))
            reqs.append(comm.Irecv(current_column_index, source=0, tag=5))
            reqs.append(comm.Irecv(up_neighbor, source=0, tag=6))
            reqs.append(comm.Irecv(previous_column, source=0, tag=7))

            for req in reqs:
                req.wait()

            print("Processing for previous column: " + str(previous_column))
            print("Up neighbor: " + str(up_neighbor))
            print("current_column_index: " + str(current_column_index))
            print("current_row_index: " + str(current_row_index))

            if current_row_index > 0:
                for i in range(len(previous_column)):
                    current_column[i] = min(previous_column[i], up_neighbor, previous_column[i - 1]) + cost_matrix[current_column_index, current_row_index]
                    up_neighbor = current_column[i]
            else:
                for i in range(len(previous_column)):
                    current_column[i] = up_neighbor + cost_matrix[i + 1, current_row_index]
                    up_neighbor = current_column[i]

            print("Finished processing current column: " + str(current_column))

            comm.Send(current_column, dest=0, tag=9)

            first_run = False

except:
    print("Breaking")
        

comm.Disconnect()