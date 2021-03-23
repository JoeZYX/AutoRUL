#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    #print(os.listdir('src'))
    rul = RemainingUsefulLife("CMAPSSData")
    rul.auto_rul(["FD002"], ["FD002"], ["FD002"])
