from rul import RemainingUsefulLife

if __name__ == '__main__':
    folds = ['FD001', 'FD002', 'FD003', 'FD004', 'FD005']

    for fold in folds:
        rul = RemainingUsefulLife("CMAPSSData")
        rul.auto_rul([fold], [fold], [fold])
        rul.test()
        print(rul.get_results())
        rul.export_to_csv(fold)


