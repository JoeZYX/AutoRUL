#import os

from rul import RemainingUsefulLife

if __name__ == '__main__':
    #print(os.listdir('src'))
    rul = RemainingUsefulLife("C:/Users/I539046/Documents/Data Science/SAP/Cahit/AL/FeedbackBoost/Data/NASA_Turbofan",
                              "nasa", ['phase', 'ts_int', 'IMM_FAILURE', 'RUL'])

    rul.train([0, 192], [193, 230])
