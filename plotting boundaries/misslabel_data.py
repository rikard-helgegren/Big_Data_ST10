import random

def misslabel_data_list(data, amount):

    # Look upp which are the labels
    if len(data[0]) > 1:
        labels = [row[-1] for row in data]
        labelSet = list(set(labels))
    else:
        labels = data
        labelSet = list(set(data))

    # Determine which to change
    if amount < 1:
        mislabel_idx = random.sample(range(0, len(labels)),
                                     round(len(labels)*amount))
    else:
        mislabel_idx = random.sample(range(0, len(labels)),
                                     round(amount))
    
    # Change the label
    for i in mislabel_idx:
        tempSet = labelSet.copy()
        tempSet.remove(labels[i])
        labels[i] = tempSet[random.randint(0, len(tempSet)-1)]

    if len(data[0]) > 1:
        for i in range(0, len(labels)):
            data[i][-1] = labels[i]
    else:
        data = labels
        
    return data
