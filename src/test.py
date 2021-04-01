#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    #print(os.listdir('src'))
    rul = RemainingUsefulLife("CMAPSSData")
    rul.auto_rul(["FD002"], ["FD002"], ["FD002"])
    rul.test()
    print(rul.get_results())
    rul.export_to_csv('FD002')
