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