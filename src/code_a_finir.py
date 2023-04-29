fig , ax = plt.subplots(figsize=(10,10))
ax1 = plt.subplot2grid(shape=(3,3), loc=(0,0), rowspan=3)
ax2 = plt.subplot2grid(shape=(3,3), loc=(0,1), rowspan=3)

plt.tight_layout()
plt.show()



fig.suptitle('{} (Response rate: {:.2f}%)'.format(temp_col, meta[(meta.index == temp_col)]['response_rate'].values[0]), fontsize=14)


df_out_Hospital = df.loc[df['TARGET'] == 0,:]
total = df_out_Hospital.groupby('CODE_GENDER')[['TARGET']].count().reset_index()
total = total.set_index('CODE_GENDER')
plt.figure(figsize=(14, 14))
total.plot(kind='bar', stacked=True, color=['red', 'skyblue']).set_title("Utilisations d'énérgies",fontsize=14)
plt.xlabel('PrimaryPropertyType')
plt.ylabel('Consommation_kBtu')
plt.show()


def bar_plot(df, feature="", bar_title=""):
    
    temp = df[feature].value_counts()
    data = pd.DataFrame(
        {'labels': temp.index,
        'values': temp.values
        })
    plt.figure(figsize = (6,6))
    plt.title(bar_title)
    sns.set_color_codes("pastel")
    sns.barplot(x = 'labels', y="values", data=data)
    locs, labels = plt.xticks()
    plt.show()

temp_col="CODE_GENDER"
cnt_srs = application_train[[temp_col, 'TARGET']].groupby([temp_col], as_index=False).mean().sort_values(by=temp_col)


def plot_category(data, col_name="", num_categories = 4, labels = None, plot_name ="",save_fig=False):
    
    """
    plot_category() : Création de catégories de la colonne col_name. 
    
    Paramètres: 
    ***********
    data : données à visualiser 
    col_name : le nom de la colonne. 
    num_categories : Nombre de catégories. par défaut : 4.
    labels : Résultat labels des clusters. 
    plot_name : Nom du graphique pour l'enregistrement. Spécifier le chemin du dossier dans lequel enregistrer le graphique. 
    save_fig : True : sauvagrder de la figure au format PNG. 
    
    
    return:
    *******
    graphique : distribution des catégories en pourcentage (%) pour chaque cluster selon la variable col_name.
    
    """
    # Copie du data original. 
    data_plot = data.copy()
    
    # Création de la colonne cluster. 
    data_plot['cluster'] = labels
    
    # Utilisez la fonction pd.cut() pour créer les catégories basées sur la colonne col_name.
    data_plot[f'{col_name}_category'] = pd.cut(data_plot[col_name], bins= num_categories, include_lowest=True , precision = 0)
    
    # groupby by = 'cluser' et col_name_category et renomage de la colonne par Total
    df_to_plot = data_plot.groupby(['cluster', f'{col_name}_category']).size().reset_index(name='Total')
    
    # Calcul du pourcentage pour chaque catégorie. 
    df_to_plot['Percentage'] = round(100* df_to_plot['Total'] / df_to_plot.groupby(['cluster'])['Total'].transform('sum'), 2)
    
    # Pivot pour traçer le graphique
    df_to_plot = df_to_plot.pivot_table('Percentage', 'cluster', f'{col_name}_category')
    
    # --- Visualisation ---
    #--Config figure------: 
    xy_label = dict(fontweight='bold', fontsize=9)
    colors = generate_colors(num_categories)
    suptitle = dict(fontweight='heavy', x=0.124, y=0.98, ha='left', fontsize=12)
    title = dict(style='italic', fontsize=8, loc='left')
    tick_params = dict(length=3, width=1, color='#CAC9CD')
    
    #----Catégories----:
    categories_ = list(df_to_plot)[:]
    
    
    #----Plot---------:
    ax = df_to_plot.plot(kind='barh', 
                         stacked=True, 
                         figsize=(9, 5), 
                         edgecolor='black', 
                         color=colors, 
                         linewidth=0.5, 
                         alpha=0.85, 
                         zorder=3)
    
    plt.ylabel('Cluster\n', **xy_label)
    for rect in ax.patches:
        width, height = rect.get_width(), rect.get_height()
        x, y = rect.get_xy()
        if width > 5:
            ax.text(x+width/2, y+height/2, '{:.1f}%'.format(width), fontsize=6, horizontalalignment='center', verticalalignment='center')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(axis='y', alpha=0, zorder=2)
    plt.grid(axis='x', which='major', alpha=0.3, color='#9B9A9C', linestyle='dotted', zorder=1)
    plt.legend(categories_, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderpad=2, frameon=False, fontsize=8, columnspacing=3)
    plt.suptitle(f'Distribution des Cluster selon : {col_name}') 
    plt.tick_params(bottom='on', **tick_params)
    for spine in ax.spines.values():
        spine.set_color('None')
    for spine in ['bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#CAC9CD')
        
        
    # Save Figure:
    if save_fig:
        plt.savefig(f"figures/EDA/plot_{plot_name}.png", transparent=True, bbox_inches='tight', dpi=200)
    plt.show()






def plot_stats(df, feature, label_rotation=False, horizontal_layout=True, traget="TARGET"):
    
    
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, feature: temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, traget]].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by=traget, ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")


    s = sns.barplot(ax=ax1, x = feature, y= feature ,data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y= traget, order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Pourcentage avec la TARGET = 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()



#stacked bar plots matplotlib: https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
def stack_plot(data, xtick, col2='TARGET', col3='total'):
    ind = np.arange(data.shape[0])
    
    if len(data[xtick].unique())<5:
        plt.figure(figsize=(5,5))
    elif len(data[xtick].unique())>5 & len(data[xtick].unique())<10:
        plt.figure(figsize=(7,7))
    else:
        plt.figure(figsize=(15,15))
    p1 = plt.bar(ind, data[col3].values)
    p2 = plt.bar(ind, data[col2].values)

    plt.ylabel('Loans')
    plt.title('Number of loans aproved vs rejected')
    plt.xticks(ticks=ind,rotation=90,labels= list(data[xtick].values))
    plt.legend((p1[0], p2[0]), ('capable', 'not capable'))
    plt.show()
    
    
def univariate_barplots(data, col1, col2='TARGET', top=False):
    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039
    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()

    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
    temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg(total='count')).reset_index()['total']
    temp['Avg'] = pd.DataFrame(data.groupby(col1)[col2].agg(Avg='mean')).reset_index()['Avg']
    
    temp.sort_values(by=['total'],inplace=True, ascending=False)
    
    if top:
        temp = temp[0:top]
    
    stack_plot(temp, xtick=col1, col2=col2, col3='total')
    print(temp.head(5))
    print("="*50)
    print(temp.tail(5))