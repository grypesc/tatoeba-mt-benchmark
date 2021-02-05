from random import shuffle


def save_set(set, data_path):
    with open(data_path + "_eng.txt", 'w') as file:
        file.writelines(["%s\n" % item[0] for item in set])
    with open(data_path + "_spa.txt", 'w') as file:
        file.writelines(["%s\n" % item[1] for item in set])


if __name__ == '__main__':
    '''Generates data sets from manythings.org dataset'''
    lines = open("../data/spa.txt", encoding='utf-8').read().strip().split('\n')
    pairs = [[s.lower() for s in l.split('\t')[0:2]] for l in lines]
    shuffle(pairs)
    train_set = pairs[0:int(0.6 * len(pairs))]
    validation_set = pairs[int(0.6 * len(pairs)):int(0.8 * len(pairs))]
    test_set = pairs[int(0.8 * len(pairs)):]

    save_set(train_set, "../data/train")
    save_set(validation_set, "../data/validation")
    save_set(test_set, "../data/test")
