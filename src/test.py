from rul import RemainingUsefulLife

if __name__ == '__main__':
    idx_values = ['10', '70', '130', '150']
    fold = "FD004"

    for i in idx_values:
        fold_idx = fold + "_WT_" + i
        rul = RemainingUsefulLife("CMAPSSData")
        rul.auto_rul([fold], [fold_idx], [fold_idx])
        rul.test()
        print(rul.get_results())
        rul.export_to_csv(fold_idx)
