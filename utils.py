import numpy as np
import matplotlib.pyplot as plt

def attention_init(train_len, test_len, attention_para, init_value):
    u_train = np.full((train_len, attention_para),
                      init_value, dtype=np.float32)
    u_test = np.full((test_len, attention_para),
                     init_value, dtype=np.float32)
    return u_train, u_test

def pltfunction(acc,loss,name):
    fig, ax = plt.subplots()
    epoch = range(len(acc))
    plt.xlabel('epoch')
    plt.plot(epoch,acc,"x-",label=name+'acc')
    plt.plot(epoch,loss, "+-", label=name+'loss')
    plt.grid(True)
    plt.legend(loc=1)
    plt.savefig(name)
    plt.show()

