#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    #print(os.listdir('src'))
    data = "FD001_WT_150"
    rul = RemainingUsefulLife("CMAPSSData")
    rul.auto_rul(["FD001"], [data], [data])
    rul.test()
    print(rul.get_results())
    rul.export_to_csv(data)
