from multiprocessing import Process, Pipe
import torch


def f(conn):
    data = {'force': [[-0.0934, -0.0214,  0.0495], [-0.0651, -0.0909, -0.0848], [-0.0042,  0.0523, -0.0092],
                      [0.0637, -0.0939, -0.0402], [0.0107,  0.0885,  0.0101]],
            'initial_state': [2, 4, 3]}
    conn.send(data)
    conn.send((23, 32, 234))
    conn.close()



if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    print(parent_conn.recv())
    p.join()
