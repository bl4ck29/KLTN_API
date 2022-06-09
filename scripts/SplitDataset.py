def SplitDataset(encoded_text, labels, train_test_split=[0.7, 0.2, 0.1]):
    fakes = encoded_text[ : labels.index(1)]
    label_fakes = labels[ : labels.index(1)]
    reals = encoded_text[labels.index(1) : ]
    label_reals = labels[ : labels.index(1)]

    data = [fakes, reals]
    labels = [label_fakes, label_reals]

    lst_len = [len(fakes), len(reals)]
    sizes = []
    for length in lst_len:
        sizes.append([int(length * i) for i in train_test_split])

    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
    for i in range(len(data)):
        size = sizes[i]
        X_train.extend(data[i][ : size[0]])
        y_train.extend(labels[i][ : size[0]])

        X_val.extend(data[i][size[0] : size[0]+size[1]])
        y_val.extend(labels[i][size[0] : size[0]+size[1]])

        X_test.extend(data[i][size[0]+size[1] : ])
        y_test.extend(labels[i][size[0]+size[1] : ])
    return X_train, X_val, X_test, y_train, y_val, y_test