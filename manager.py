from mpi4py import MPI
import numpy as np
import sys

def matrix(value, M, N):

    matrix = np.zeros((M, N), np.int64)
    return matrix

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['agent.py'], maxprocs=2)
rank = comm.Get_rank()
size = comm.Get_size()

A = [(2,2), (0,4), (2,6), (4,5), (4,7), (5,7)]
B = [(1,1), (0,6), (4,4), (4,5), (5,5)]

M = len(A)
N = len(B)

m_n = np.array((M,N))
print("About to send matrix dimensions (M x N): " + str(m_n))
comm.Bcast(m_n, root = MPI.ROOT)

# Manager Process initializes matrices.
cost_matrix = matrix(0, M, N) # Initialize with zeros

for i in range(M):
    for j in range(N):
        cost_matrix[i,j] = (A[i][0] - B[j][0])**2 + (A[i][1] - B[j][1])**2 # distance function

print(cost_matrix)

dtw_matrix = matrix(0, M, N) # Initialize with zeros
dtw_matrix[0, 0] = cost_matrix[0, 0] # Initialize top left cell

neighbor = dtw_matrix[0, 0]
# COLUMN
for i in range(1, M):
    dtw_matrix[i, 0] = neighbor + cost_matrix[i, 0]
    neighbor = dtw_matrix[i, 0]
# ROW
neighbor = dtw_matrix[0, 0]
for i in range(1, N):
    dtw_matrix[0, i] = neighbor + cost_matrix[0, i]
    neighbor = dtw_matrix[0, i]

print(dtw_matrix)

send_row, send_column = True, True
req_for_row, req_for_col = [], []

row_counter, col_counter = 1, 1
row_limit, col_limit = M, N

complete = False

req_for_agent0, req_for_agent1 = None, None

while not complete:
    
    if send_row == True:
        print(f"sending row {row_counter}")
        # Stuff that we need to send to the row process
        
        previous_row = np.array(dtw_matrix[row_counter - 1, col_counter - 1:], dtype=np.int64)
        cost_row = np.array(cost_matrix[row_counter, col_counter:], dtype=np.int64)
        left_neighbor = np.array(dtw_matrix[row_counter, col_counter - 1], dtype=np.int64)

        req_for_row.append(comm.Isend(np.array(len(previous_row), dtype=np.int64), dest=0, tag=1))
        req_for_row.append(comm.Isend(previous_row, dest=0, tag=2))
        req_for_row.append(comm.Isend(left_neighbor, dest=0, tag=3))

        new_row = np.zeros(len(previous_row) - 1, dtype = np.int64)
        req_for_row.append(comm.Isend(cost_row, dest=0, tag=4))
        send_row = False

    if send_column == True:
        print(f"sending col {col_counter}")
        previous_col = np.array(dtw_matrix[row_counter - 1:, col_counter - 1], dtype=np.int64)
        cost_col = np.array(cost_matrix[row_counter:, col_counter], dtype=np.int64)
        up_neighbor = np.array(dtw_matrix[row_counter - 1, col_counter], dtype=np.int64)

        req_for_col.append(comm.Isend(np.array(len(previous_col), dtype=np.int64), dest=1, tag=1))
        req_for_col.append(comm.Isend(previous_col, dest=1, tag=2))
        req_for_col.append(comm.Isend(up_neighbor, dest=1, tag=3))

        new_col = np.zeros(len(previous_col) - 1, dtype = np.int64)
        req_for_col.append(comm.Isend(cost_col, dest=1, tag=4))
        send_column = False
    
    if len(req_for_row) > 0:
        if MPI.Request.Testall(req_for_row) == True:
            print(f"request for agent to rcv row completed")
            req_for_agent0 = comm.Irecv(new_row, source = 0, tag=5)
            req_for_row = []
    if len(req_for_col) > 0:
        if MPI.Request.Testall(req_for_col) == True:
            print(f"request for agent to rcv col completed")
            req_for_agent1 = comm.Irecv(new_col, source = 1, tag=6)
            req_for_col = []

    if req_for_agent0 != None:
        if MPI.Request.Test(req_for_agent0) == True:
            print(f"Manager recvd row: {new_row}")
            for i in range(len(new_row)):
                dtw_matrix[row_counter, col_counter + i] = new_row[i]
            row_counter += 1
            if row_counter < row_limit:
                send_row = True
            else:
                complete = True
            req_for_agent0 = None
            print(dtw_matrix)
            print(f"Computed:: Rows: {row_counter}, Cols: {col_counter}")
    if req_for_agent1 != None:
        if MPI.Request.Test(req_for_agent1) == True:
            print(f"Manager recvd col: {new_col}")
            for i in range(len(new_col)):
                dtw_matrix[row_counter + i, col_counter] = new_col[i]
            col_counter += 1
            if col_counter < col_limit:
                send_column = True
            else:
                complete = True
            req_for_agent1 = None
            print(dtw_matrix)
            print(f"Computed:: Rows: {row_counter}, Cols: {col_counter}")



for req in req_for_row + req_for_col:
    req.Cancel()

comm.Send(np.array(1, dtype=np.int64), dest=0, tag=7)
print("Sent sigterm to 0")
comm.Send(np.array(1, dtype=np.int64), dest=1, tag=7)
print("Sent sigterm to 1")

print("RESULT:\n" + str(dtw_matrix))

# comm.Disconnect()
sys.exit(0)