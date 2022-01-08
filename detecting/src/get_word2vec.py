import socket
import gensim
import numpy as np
import os

def get_word2vec(text, model):

    myvector = np.zeros(shape = (100))

    for word in text.split(' '):
        try:
            myvector += model.wv[word]
        except:
            continue

    if str(myvector[0]) == '0.0' and str(myvector[1]) == '0.0'and  str(myvector[2]) == '0.0':
        myvector = np.full((100), 0.5)
    
    return myvector

if __name__ == '__main__':

    word2vec_porn = gensim.models.Word2Vec.load('/home/runner/work/BroBand/BroBand/detecting/model/word2vec/porn_word2vec.model')
    
    # создание сокета для TCP/IP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Привязываем сокет к порту
    server_address = ('localhost', 9090)
    sock.bind(server_address)

    # слушаем
    sock.listen(1)
    
    print("Service is ready...")

    # ждем соединения
    while True:

        connection, addr = sock.accept()

        try:
            while True:

                data = connection.recv(1024)
                print(data.decode())

                if data:
                    vec = ' '.join(list(map(str, get_word2vec(str(data.decode()), word2vec_porn))))
                    connection.sendall(vec.encode('utf-8'))
                    print(vec)
                    del vec
                    del data
                else:
                    break

        finally:
            connection.close()
