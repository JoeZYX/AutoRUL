#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    i_values = ['10', '70', '130', '150']
    s = "FD004"
    for i in i_values:
        si = s + "_WT_" + i
        rul = RemainingUsefulLife("CMAPSSData")
        rul.auto_rul([s], [si], [si])
        rul.test()
        print(rul.get_results())
        rul.export_to_csv(si)
