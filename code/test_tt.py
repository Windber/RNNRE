from task.tomitatrain import *
if __name__ == '__main__':
    #configname = sys.argv[1]
    configname = "Tomita1Config"
    config = globals()[configname]
    start = time.time()
    t = TaskForRNN(config)
    t.train()
    t.test()
    print("time: %f" % (time.time()-start))
