def thresholdLikes(df, UIDmin, MIDmin):
    """
    Removes from data set those items having fewer than


    """

    userCt = df.uid.unique().shape[0]
    itemCt = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(userCt*itemCt) * 100
    print("Starting likes info")
    print("Number of users: {userCt}")
    print("Number of models: {itemCt}")
    print("Sparsity: {sparsity:4.3f}%")

    done = False
    while not done:
        starting_shape = df.shape[0]
        mid_counts = df.groupby('uid').mid.count()
        df = df[~df.uid.isin(mid_counts[mid_counts < MIDmin].index.tolist())]
        uid_counts = df.groupby('mid').uid.count()
        df = df[~df.mid.isin(uid_counts[uid_counts < UIDmin].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    assert(df.groupby('uid').mid.count().min() >= MIDmin)
    assert(df.groupby('mid').uid.count().min() >= UIDmin)

    userCt = df.uid.unique().shape[0]
    itemCt = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(userCt*itemCt) * 100
    print('Ending likes info')
    print('Number of users: {}'.format(userCt))
    print('Number of models: {}'.format(itemCt))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    return df
