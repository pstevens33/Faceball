import pandas as pd
import numpy as np

df_war = pd.read_json('../data/pitcher_war.json', lines=True)
df_war.loc[df_war['years_of_service'] == 0, 'years_of_service'] = 1
df_war['avg_war'] = round(df_war['war'] / df_war['years_of_service'], 0)
df_war.loc[df_war['avg_war'] < 0, 'avg_war'] = 0
df = pd.read_pickle('../data/recognized_faces_df')
#df_war = df_war[df_war['avg_war'] > 0.5]
original_names = df['name'].values
new_names = df_war['name'].values
avg_wars = df_war['avg_war'].values
years_of_service = df_war['years_of_service'].values
names = []
wars = []
for i, name in enumerate(original_names):
    for j, name2 in enumerate(new_names):
        if name == new_names[j]:
            names.append(new_names[j])
            wars.append(avg_wars[j])
            break

wars = np.array(wars)
np.save('../data/high_y_wars', wars)
