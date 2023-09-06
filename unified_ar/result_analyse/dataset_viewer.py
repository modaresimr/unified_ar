from matplotlib.pylab import plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.patches as patches

import plotly.figure_factory as ff

import matplotlib._color_data as mcd
import matplotlib.dates as mdates
from IPython.display import display


def displaycontent(dataset):
    if not (hasattr(dataset, 'sensor_events')):
        return
    print('sensor events:')
    display(dataset.sensor_events.iloc[20:25])
    print('activity_events:')
    display(dataset.activity_events.loc[1:1])
    print('sensor_desc:')
    display(dataset.sensor_desc.iloc[1:3])
    print("Activites: ", dataset.activities)
    for a, v in dataset.activities_map.items():
        items = dataset.activity_events.loc[dataset.activity_events['Activity'] == a]['Duration']
        # print(a,v)
        # display(items.describe())
        print(a, v, '\t--> count=', items.count(), ' avg duration=', str(items.mean()))
    x = dataset.activity_events.copy()
    x['Duration'] = x['Duration'].dt.seconds
    x.boxplot(by='Activity', column='Duration')


# loadA4HDataSet()
# loadVanKasterenDataset()
# loadKaryoAdlNormalDataset();
# display()


def view(dataset, i):
    if not (hasattr(dataset, 'sensor_events')):
        return
    tmp_act_evants = dataset.activity_events.loc[dataset.activity_events['Activity'] == i]

    print(dataset.activities_map[i])
    print(tmp_act_evants['Duration'].describe())
    if len(tmp_act_evants) == 0:
        return

    fig = plt.figure()

    tmp_act_evants['StartTime'].iloc[0]
    all = pd.DataFrame()
    for index, row in tmp_act_evants.iterrows():
        myse = dataset.sensor_events.loc[(dataset.sensor_events['time'] >= row['StartTime']) & (dataset.sensor_events['time'] <= row['EndTime'])].copy()
        myse['relative'] = dataset.sensor_events['time'] - row['StartTime']
        myse['hit time'] = myse['relative'] / row['Duration']
        all = pd.concat([all, myse[['hit time', 'SID']]])
        # plt.scatter(myse['hit time'],myse['SID'])

    tmp = all.copy()

    tmp['hit time'] = (tmp['hit time'] * 2).round(0) / 2
    fig = plt.figure(figsize=(10, 5))
    a = pd.pivot_table(tmp, columns='hit time', index='SID', aggfunc=np.count_nonzero, fill_value=0)
    a = a / a.max()
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    ax = plt.axes()
    sns.heatmap(a / a.max(), cmap=sns.cm.rocket_r, ax=ax)
    ax.set_title(dataset.activities_map[i])


# view(5)


def plotAct(dataset, acts):
    firstacts = acts.iloc[0]
    acts = acts.loc[acts['StartTime'] < firstacts['StartTime'] + pd.Timedelta('7d')]
    lastact = acts.iloc[-1]
    lastactinDay = acts.loc[acts['StartTime'] < firstacts['StartTime'] + pd.Timedelta('20h')].iloc[-1]

    # for a in dataset.activities:
    #     acts = acts.append({
    #         'Activity': dataset.activities_map_inverse[a],
    #         'StartTime': firstacts['StartTime'],
    #         'EndTime': firstacts['StartTime']
    #     },
    #                        ignore_index=True)

    acts = acts.sort_values(by='Activity')

    df2 = acts.apply(lambda x: dict(Task=dataset.activities_map[x.Activity], Color=0, Start=x.StartTime, Finish=x.EndTime), axis=1).tolist()
    # configure_plotly_browser_state()
    # init_notebook_mode(connected=False)
    # fig=ff.create_gantt(df2, index_col='Color', group_tasks=True)

    fig = ff.create_gantt(df2, group_tasks=True)
    fig['layout'].update(margin=dict(l=150))
    fig['layout'].update(xaxis=dict(range=[firstacts['StartTime'], lastactinDay['EndTime']],
                                    rangeselector=dict(buttons=list([
                                        dict(count=4, label='4h', step='hour', stepmode='backward'),
                                        dict(count=6, label='6h', step='hour', stepmode='backward'),
                                        dict(count=8, label='8h', step='hour', stepmode='backward'),
                                        dict(count=10, label='10h', step='hour', stepmode='backward'),
                                        dict(count=12, label='12h', step='hour', stepmode='backward'),
                                        dict(count=1, label='1d', step='day', stepmode='backward'),
                                        dict(count=5, label='5d', step='day', stepmode='backward'),
                                        dict(step='all')
                                    ])),
                                    rangeslider=dict(
                                        visible=True,
                                        range=[firstacts['StartTime'], lastact['EndTime']],
                                    )))

    fig.show()


def sensor_hitmap(dataset):
    if not (hasattr(dataset, 'sensor_events')):
        return
    actscount = len(dataset.activities)
    import matplotlib.pyplot as plt

    fig, subplots = plt.subplots((actscount - 1) // 4 + 1, 4, sharex=True, sharey=True,figsize=(10,12))
    subplots = subplots.reshape(-1)
    for i in dataset.activities_map:
        tmp_act_evants = dataset.activity_events.loc[dataset.activity_events['Activity'] == i]

        # print(dataset.activities_map[i])
        # print(tmp_act_evants['Duration'].describe())
        if len(tmp_act_evants) == 0:
            continue

        # fig = plt.figure()

        # tmp_act_evants['StartTime'].iloc[0]
        all = pd.DataFrame()
        for index, row in tmp_act_evants.iterrows():
            myse = dataset.sensor_events.loc[(dataset.sensor_events['time'] >= row['StartTime']) & (dataset.sensor_events['time'] <= row['EndTime'])].copy()
            myse['relative'] = dataset.sensor_events['time'] - row['StartTime']
            myse['hit time'] = myse['relative'] / row['Duration']
            all = pd.concat([all, myse[['hit time', 'SID']]])
            # plt.scatter(myse['hit time'],myse['SID'])

        tmp = all.copy()

        tmp['hit time'] = (tmp['hit time'] * 2).round(0) / 2
        fig = plt.figure(figsize=(10, 5))
        a = pd.pivot_table(tmp, columns='hit time', index='SID', aggfunc=np.count_nonzero, fill_value=0)
        a = a / a.max()
        # plt.imshow(a, cmap='hot', interpolation='nearest')
        ax = subplots[i]
        sns.heatmap(a / a.max(), cmap=sns.cm.rocket_r, ax=ax)
        ax.set_title(dataset.activities_map[i])
