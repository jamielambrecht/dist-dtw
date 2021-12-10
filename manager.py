from mpi4py import MPI
import numpy as np
import sys

def matrix(value, M, N):

    matrix = np.zeros((M, N), np.int64)
    return matrix

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['agent.py'], maxprocs=2)
rank = comm.Get_rank()
size = comm.Get_size()
print("Remote group size:" + str(comm.remote_group.size))
print("This program requires exactly 3 processes, one manager and two agents. " + "I am the manager.")

A = [(2,2), (0,4), (2,6), (4,5)]
B = [(1,1), (0,6), (4,4)]

M = len(A)
N = len(B)

m_n = np.array((M,N))
print("About to send matrix dimensions (M x N): " + str(m_n))
comm.Bcast(m_n, root=MPI.ROOT)
    
# Manager Process initializes matrices.
cost_matrix = matrix(0, M, N) # Initialize with zeros

for i in range(M):
    for j in range(N):
        cost_matrix[i,j] = (A[i][0] - B[j][0])**2 + (A[i][1] - B[j][1])**2 # distance function

comm.Bcast(cost_matrix, root=MPI.ROOT)

dtw_matrix = matrix(0, M, N) # Initialize with zeros
dtw_matrix[0, 0] = cost_matrix[0, 0] # Initialize top left cell

current_row_index, current_column_index = np.array(0), np.array(0)
current_row = np.array(dtw_matrix[0, 1:])
current_column = np.array(dtw_matrix[1:, 0])

update_A, update_B = True, True
complete = False
req_from_A, req_from_B = [], []

first_run = True


while not complete:

    if update_A:
        # Stuff that we need to send to the row process
        if not first_run:
            left_neighbor = dtw_matrix[current_row_index, current_column_index - 1]
        else:
            left_neighbor = dtw_matrix[0, 0]
        req_from_A.append(comm.Isend(current_row_index, dest=0, tag=0))
        req_from_A.append(comm.Isend(current_column_index, dest=0, tag=1))
        req_from_A.append(comm.Isend(left_neighbor, dest=0, tag=2))
        req_from_A.append(comm.Isend(current_row, dest=0, tag=3))
        if not first_run:
            current_row = np.zeros(N-current_column_index - 1, dtype=np.int64)
        else:
            current_row = np.zeros(N-current_column_index - 1, dtype=np.int64)
        
        print("Length of returned row: " + str(len(current_row)))
        
        update_A = False
    if update_B:
        # Stuff that we need to send to the column process
        if not first_run:
            up_neighbor = dtw_matrix[current_row_index - 1, current_column_index]
        else:
            up_neighbor = dtw_matrix[0, 0]
        req_from_B.append(comm.Isend(current_row_index, dest=1, tag=4))
        req_from_B.append(comm.Isend(current_column_index, dest=1, tag=5))
        req_from_B.append(comm.Isend(up_neighbor, dest=1, tag=6))
        req_from_B.append(comm.Isend(current_column, dest=1, tag=7))
        if not first_run:
            current_column = np.zeros(M-current_row_index - 1, dtype=np.int64)
        else:
            current_column = np.zeros(M-current_row_index - 1, dtype=np.int64)
        
        print("Length of returned column: " + str(len(current_column)))
        
        update_B = False

    for req in req_from_A + req_from_B:
        req.wait()

    current_row_index += 1
    current_column_index += 1

    comm.Recv(current_row, source = 0, tag=8)
    comm.Recv(current_column, source = 1, tag=9)

    print("Manager received: \n" + str(current_row) + "\n" + str(current_column))
    # print("Length of current row: " + str(len(current_row)))
    if len(current_row) != 0:
        for i in range(len(current_row)):
            dtw_matrix[current_row_index - 1, current_column_index + i] = current_row[i]
        
        update_A = True
    if len(current_row) != 0:
        for i in range(len(current_column)):
            dtw_matrix[current_row_index + i, current_column_index - 1] = current_column[i]

        update_B = True

    current_column = np.array(dtw_matrix[current_row_index:, current_column_index - 1])
    current_row = np.array(dtw_matrix[current_row_index - 1, current_column_index:])

    print("DTW Matrix:")
    print(dtw_matrix)

    print("Current row " + str(current_row))
    print("Current column " + str(current_column))

    first_run = False



comm.Disconnect()