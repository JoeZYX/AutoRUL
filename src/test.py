#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    #print(os.listdir('src'))
    rul = RemainingUsefulLife("TurbofanData",
                              non_features=["ts_int", "cnt_per_rtf", "phase", "RUL", "RUL_pw", "IMM_FAILURE"])
    rul.auto_rul(["FD001"], ["FD001"], ["FD001"])
    rul.test()
    print(rul.get_results())
    rul.export_to_csv('FD001')
