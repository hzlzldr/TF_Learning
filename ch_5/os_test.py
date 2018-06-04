#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = lanzili
__mtime__ = '2018/6/3'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
            ┃   ☃    ┃
            ┃ ┳┛  ┗┳ ┃
            ┃   ┻    ┃
            ┗━┓    ┏━┛
              ┃    ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃  永无BUG！ ┏┛
              ┗┓┓┏━┳┓┏┛
               ┃┫┫ ┃┫┫
               ┗┻┛ ┗┻┛

when I wrote this,only God and I understood what I was doing.
Now,God only knows.

"""

from multiprocessing import Process
import os
from multiprocessing import Pool
from multiprocessing import Pipe
import time
import numpy as np
"""
    win环境下，没有fork()，可以利用multiprocessing的Process类来创建子进程
"""

def run_proc(name):
    print("name is {0} and pid is {1}".format(name,os.getpid()))

def run_proc_main():
    print("ParentProcess pid is {}".format(os.getpid()))

    """
    创建子进程时，只需要传入一个执行函数和函数的参数，创建一个Process实例，用start()方法启动

    join()方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步。
    """
    p=Process(target=run_proc,args=('test',))
    print("ChildProcess start to run...")
    p.start()
    p.join()#等待进程的结束
    print("ChildProcess is over!")

def pool_test(name):
    """
    如果要启动大量的子进程，可以用进程池的方式批量创建子进程
    """
    print("run task {0} and pid is {1}".format(name,os.getpid()))
    start_time=time.time()
    time.sleep(np.random.rand()*3)
    print("Task {0} runs {1}".format(name,time.time()-start_time))

def pool_test_main():
    print("Parent id is {}".format(os.getpid()))
    p=Pool(4)#创建含有4个子进程的进程池
    #默认参数代表电脑的核数
    for i in range(5):
        p.apply(pool_test,args=(i,))

    print("Waitting for all subprocesses!")
    p.close()
    p.join()
    print("All subprocesses is over!")

    """
    对Pool对象调用join()方法会等待所有子进程执行完毕，调用join()之前必须先调用close()，
    调用close()之后就不能继续添加新的Process了
    """

def subprocess_test():
    import subprocess
    """
        当子进程不是自身本身而是一个外部进程时（可能还涉及到输入输出等操作），可以调用subprocess类来解决
    """
    print("$ nslookup www.sysu.edu.cn")
    ret=subprocess.call(['nslookup','www.sysu.edu.cn'])
    print("Exit code:%d"%ret)
    subprocess.call(['python','picture_process_test.py'])

from multiprocessing import Queue

def write(queue_name):

    print("Process start to write {}".format(os.getpid()))
    for i in range(5):
        print("Put {} to queue".format(i))
        queue_name.put(i)
        time.sleep(np.random.rand()*3)

def read(queue_name):
    print("Process start to read {}".format(os.getpid()))
    while 1:
        value=queue_name.get()
        print("from the queue get the {}".format(value))
        time.sleep(np.random.rand()*3)

def write_read_main():
    q=Queue()
    pw=Process(target=write,args=(q,))
    pr=Process(target=read,args=(q,))

    pw.start(),pr.start()

    pw.join()
    pr.terminate()#由于是无线循环，只能强制终止



def clock(interval):
    while 1:
        print("The time is %s"%time.ctime())
        time.sleep(interval*0.5)

def clock_main():
    interval=10
    p=Process(target=clock,args=(interval,))#args传入的是元组
    p.start()
    time.sleep(interval)
    p.terminate()


class Clock_Process(Process):
    """
    继承Process
    """
    def __init__(self,interval):
        Process.__init__(self)
        self.interval=interval
    def run(self):
        while 1:
            print("The time is %s"%time.ctime())
            time.sleep(self.interval*0.5)

def clock_process_main():
    p = Clock_Process(10)
    p.start()
    time.sleep(20)
    p.terminate()

def consumer(input_queue):
    while 1:
        value=input_queue.get()
        if value is None:
            break#设置哨兵，当生产队列结束后，退出消费队列
        print(value)
    print("It's done!")

def producer(sequence,output_queue):
    for item in sequence:
        output_queue.put(item)

def consumer_producer_main():
    """
    Queue实现进程间通信（Interprocess Communication，IPC）
    Pipe也可以
    """
    q=Queue()
    sequence=['a','b','c']

    cons_q=Process(target=consumer,args=(q,))
    cons_q.start()

    producer(sequence,q)
    q.put(None)#发送哨兵(sentinel)
    cons_q.join()


def pipe_consumer(pipe):
    output_queue,input_queue=pipe
    input_queue.close()#关闭输入端，因为用不到双向通信，避免该端口处于挂起状态

    while 1:
        try:
            value=output_queue.recv()#注意获取函数不同于队列
            print(value)
        except EOFError:
            break

    print("Recv is over!")

def pipe_producer(sequence,input_queue):

    for item in sequence:
        input_queue.send(item)#管道用的是send

def pipe_consumer_main():
    (output_queue,input_queue)=Pipe()

    cons_q=Process(target=pipe_consumer,args=((output_queue,input_queue),))#记得输入元组参数
    cons_q.start()

    output_queue.close()#关闭生产者输出管道

    sequence=[i**2 for i in range(4)]
    pipe_producer(sequence,input_queue)

    #关闭管道，表示完成
    input_queue.close()

    cons_q.join()#等待消费者的完成
    """
        管道是由os进行引用计数的，必须在所有进程中关闭管道才能生成EOFError
        因此在生产者中关闭管道不会有任何效果，除非消费者也关闭了相同的管道端点
    """

def adder(pipe):
    server_p,client_p=pipe

    client_p.close()

    while 1:
        try:
            (x,y)=server_p.recv()
        except EOFError:
            break
        value=x+y
        server_p.send(value)

    print("Server done!")

def adder_main():
    server_p,client_p=Pipe()

    server=Process(target=adder,args=((server_p,client_p),))
    server.start()

    #关闭客户端中的服务器通道
    server_p.close()

    client_p.send((4,5))#客户端发送请求
    print(client_p.recv())

    client_p.send(("hello ","pipe"))
    print(client_p.recv())

    #完成，关闭通道
    client_p.close()

    #等待消费者关闭进程
    server.join()


if __name__ == '__main__':

    try:
        #run_proc_main()
        #pool_test_main()
        #subprocess_test()
        #write_read_main()
        #clock_main()
        #consumer_producer_main()
        #pipe_consumer_main()
        adder_main()
    except Exception as e:
        print(e)
    finally:
        print("os-owo-so")

