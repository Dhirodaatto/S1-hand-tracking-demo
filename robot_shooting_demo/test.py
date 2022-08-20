from multiprocessing import Process, Value

import threading

def thread():
    global t
    t.value = 10

def proc(t):
    t.value = 100

if __name__ == '__main__':
    t = Value("d", 0)
    t.value = 2
    
    print(f'original t value = {t.value}')
    
    x = threading.Thread(target = thread, args = ())
    
    y = Process(target = proc, args = (t,))
    
    x.start()
    x.join()
    
    print(f'after thread exec t = {t.value}')
    
    y.start()
    y.join()
    
    print(f'after process exec t = {t.value}')