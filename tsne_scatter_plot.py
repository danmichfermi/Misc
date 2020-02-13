#Parts of this was taken from an online resource which works well
def tsnescatterplot(model, words, name):
#initializing arrays
    wor_test=model.wv.__getitem__(words[0])
    arrays = np.empty((0, len(wor_test)), dtype='f')
    word_labels = []
    color_list  = []
    
    # adds the vector for each of the closest words to the array
    for wrd_score in words:
        wrd_vector = model.wv.__getitem__([wrd_score])
        if words in word_labels:
        	color_list.append('red')
        else:
        	color_list.append('blue')
        word_labels.append(words)
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # PCA dimensionality reduction
    reduc = PCA(n_components=15).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
        # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    titlestr='t-SNE visualization of vectorized'+name
    plt.title(titlestr)
    path=name+"tSNEvec.pdf"
    plt.savefig(path)

