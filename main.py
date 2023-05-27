import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


# 环境和参数的配置
def prepare_profile():
    mpl.rcParams['figure.figsize'] = (12, 6)  # 修改图片大小
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.sans-serif'] = ['STXIHEI']
    plt.rcParams['axes.unicode_minus'] = False


# 全球用户每年的各项数据的分析与可视化
def global_internet_users_analysis():
    # 全球每年的互联网用户总数分析与可视化：
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2, 2, 1)
    year_groups = global_users.groupby('Year')
    internet_users_groups = year_groups['No. of Internet Users']
    internet_users_sum = internet_users_groups.sum()
    plt.title('全球每年互联网用户总数')
    plt.xlabel('年份')
    plt.ylabel('全球每年互联网用户总数')
    plt.bar(internet_users_sum.index, internet_users_sum.values, color='cornflowerblue', width=0.6)
    plt.plot(internet_users_sum, label='No. of Internet Users', lw=3, color='cornflowerblue')

    # 全球每年每100人移动端互联网订阅数、互联网使用人数比例、每100人宽带订阅数的平均值分析与可视化：
    year_groups = global_users.groupby('Year')
    title_mapper = {'Cellular Subscription': '全球每年每100人移动端互联网订阅数',
                    'Internet Users(%)': '全球每年互联网使用人数比例',
                    'Broadband Subscription': '全球每年每100人宽带订阅数的平均值'}
    i = 2
    for column in ['Cellular Subscription', 'Internet Users(%)', 'Broadband Subscription']:
        plt.subplot(2, 2, i)
        i += 1
        internet_users_groups = year_groups[column]
        internet_users_sum = internet_users_groups.mean()
        column_max = internet_users_groups.max()
        plt.title(title_mapper[column])
        plt.xlabel('年份')
        plt.ylabel(title_mapper[column])
        plt.plot(column_max, label=column + ' max', lw=2, linestyle=(0, (5, 1)))
        plt.plot(internet_users_sum, label=column + ' mean', lw=3, linestyle=(0, (1, 1)))
        plt.legend(loc='upper right')
    plt.show()


# 获取2020年国家地区数据封装成的DataFrame
def get_2020_entities_dataframe():
    entity_group = global_users.groupby('Entity')
    entity_2020_df = pd.DataFrame()
    for entity, entity_df in entity_group:
        if entity == 'World':
            continue
        entity_2020 = entity_df[entity_df['Year'] == 2020]
        entity_2020_df = pd.concat([entity_2020_df, entity_2020], join='outer', axis=0)
    return entity_2020_df.set_index('Entity')


# 2020年各个国家地区的用户占比饼图绘制
def entities_2020_internet_users_percentage_pie():
    entity_2020_df = get_2020_entities_dataframe()
    entity_2020_df['No. of Internet Users'] /= entity_2020_df['No. of Internet Users'].sum()

    # 只筛选用户数量最多的10组数据，其他数据用`other`代替
    entity_2020_df.sort_values(by='No. of Internet Users', axis=0, ascending=False, inplace=True)
    other = entity_2020_df.iloc[10:].loc[:, 'No. of Internet Users'].sum()
    other_df = pd.DataFrame(data={'': {'Entity': 'Other', 'No. of Internet Users': other}}).T
    other_df.set_index('Entity', inplace=True)
    processed_data = pd.concat([entity_2020_df.head(10), other_df], axis=0, join='outer')

    # 绘制饼图
    explode_arr = np.zeros(shape=(11))
    explode_arr[0] = 0.07
    plt.axes(aspect=1)
    plt.title('2020年各个国家地区的互联网用户占比')
    plt.pie(processed_data['No. of Internet Users'], labels=processed_data.index, explode=explode_arr,
            labeldistance=1.1, autopct='%2.1f%%', pctdistance=0.9, shadow=True)
    plt.legend(loc='lower right', bbox_to_anchor=(0.5, 0., 1.0, 0.5), ncols=2)
    plt.show()

    # 绘制柱状图
    bar_dict = dict(zip(processed_data.index, processed_data['No. of Internet Users']))
    plt.title('2020年各个国家地区的互联网用户占比')
    plt.bar(bar_dict.keys(), bar_dict.values(), color='blue')
    plt.show()


# 2020年各国家地区互联网用户占比分布直方图
def entities_2020_internet_users_percentage_distribution_histogram():
    entity_2020_df = get_2020_entities_dataframe()
    internet_users_percentage_sr = entity_2020_df['Internet Users(%)']
    plt.title('2020年各国家地区互联网用户占比分布直方图')
    plt.xlabel('互联网用户占比占比')
    plt.ylabel('国家地区数量')
    plt.grid()
    plt.hist(internet_users_percentage_sr, color='green', rwidth=0.6, alpha=0.75)
    plt.show()


# 2020年个国家地区互联网用户占比和移动互联网订阅量的散点图
def entities_2020_internet_users_percentage_distribution_scatter():
    entity_2020_df = get_2020_entities_dataframe()
    internet_users_percentage_sr = entity_2020_df['Internet Users(%)']
    cellular_subscription_sr = entity_2020_df['Cellular Subscription']
    plt.title('2020年个国家地区互联网用户占比和移动互联网订阅量散点图')
    plt.xlabel('互联网用户占比占比')
    plt.ylabel('移动互联网订阅量')
    plt.grid()
    plt.scatter(internet_users_percentage_sr, cellular_subscription_sr, color='blue')
    # plt.show()

    # 利用线性回归分析两者关系
    poly_reg = PolynomialFeatures(degree=2)
    x = entity_2020_df[['Internet Users(%)']]
    x_m = poly_reg.fit_transform(x)

    model_2 = linear_model.LinearRegression()
    model_2.fit(x_m, entity_2020_df[['Cellular Subscription']])
    plt.plot(x, model_2.predict(x_m))
    plt.show()


# 用每一年互联网用户的比例最大的国家地区名生成词云
def draw_internet_users_percentage_annual_top_3_wordcloud():
    text = ''
    year_groups = global_users.groupby('Year')
    for year, year_df in year_groups:
        year_df.sort_values(by='Internet Users(%)', ascending=False, inplace=True)
        top_3 = year_df.head(3)
        entities = top_3['Entity']
        for entity in entities:
            if len(entity.split()) > 1:
                text += entity.replace(' ', '_') + ' '
            else:
                text += entity + ' '
    wc = WordCloud(max_words=100, width=800, height=400, background_color='White',
                   max_font_size=150, stopwords=STOPWORDS, margin=5, scale=1.5)
    wc.generate(text)
    plt.imshow(wc)
    plt.axis("off")
    wc.to_file('wordcloud.png')
    plt.show()


# 对中国互联网用户数据的分析与可视化
def chinese_users_analysis():
    # 绘制各项指标的数值图
    plt.title('中国互联网用户的数量（单位：千万人）、占人口的比例、移动互联网订阅每一百人比例、宽带每一百人订阅比例')
    plt.xlabel('年份')
    plt.ylabel('数值')
    plt.plot(chinese_users['Year'], chinese_users['No. of Internet Users'] / 10000000, lw=4, label='数量（单位：千万人）')
    plt.plot(chinese_users['Year'], chinese_users['Internet Users(%)'], lw=4, label='占人口的比例')
    plt.plot(chinese_users['Year'], chinese_users['Cellular Subscription'], lw=4, label='移动互联网订阅每一百人比例')
    plt.plot(chinese_users['Year'], chinese_users['Broadband Subscription'], lw=4, label='宽带每一百人订阅比例')
    plt.legend(loc='upper left')
    plt.show()

    # 绘制各项指标的增长率图
    chinese_users.loc[:, 'increase of No. of Internet Users'] = 0
    chinese_users.loc[:, 'increase of Internet Users(%)'] = 0
    chinese_users.loc[:, 'increase of Cellular Subscription'] = 0
    chinese_users.loc[:, 'increase of Broadband Subscription'] = 0
    rows = len(chinese_users.index)
    for i in range(rows - 1):
        chinese_users.loc[:,'increase of No. of Internet Users'].iloc[i + 1] = 0 if chinese_users.iloc[i]['No. of Internet Users'] == 0 \
            else (chinese_users.iloc[i + 1].loc['No. of Internet Users'] - chinese_users.iloc[i]['No. of Internet Users']) / \
                 chinese_users.iloc[i]['No. of Internet Users']
        chinese_users.loc[:,'increase of Internet Users(%)'].iloc[i + 1] = 0 if chinese_users.iloc[i]['Internet Users(%)'] == 0 \
            else (chinese_users.iloc[i + 1]['Internet Users(%)'] - chinese_users.iloc[i]['Internet Users(%)']) / \
                 chinese_users.iloc[i]['Internet Users(%)']
        chinese_users.loc[:,'increase of Cellular Subscription'].iloc[i + 1] = 0 if chinese_users.iloc[i]['Cellular Subscription'] == 0 \
            else (chinese_users.iloc[i + 1]['Cellular Subscription'] - chinese_users.iloc[i]['Cellular Subscription']) / \
                 chinese_users.iloc[i]['Cellular Subscription']
        chinese_users.loc[:,'increase of Broadband Subscription'].iloc[i + 1] = 0 if chinese_users.iloc[i]['Broadband Subscription'] == 0 \
            else (chinese_users.iloc[i + 1]['Broadband Subscription'] - chinese_users.iloc[i]['Broadband Subscription']) / \
                 chinese_users.iloc[i]['Broadband Subscription']
    plt.title('中国互联网用户的数量（单位：千万人）、占人口的比例、移动互联网订阅每一百人比例、宽带每一百人订阅比例的增长率')
    plt.xlabel('年份')
    plt.ylabel('数值')
    plt.plot(chinese_users['Year'], chinese_users['increase of No. of Internet Users'] / 10000000, lw=4,
             label='数量（单位：千万人）增长率')
    plt.plot(chinese_users['Year'], chinese_users['increase of Internet Users(%)'], lw=4, label='占人口的比例')
    plt.plot(chinese_users['Year'], chinese_users['increase of Cellular Subscription'], lw=4,
             label='移动互联网订阅每一百人比例增长率')
    plt.plot(chinese_users['Year'], chinese_users['increase of Broadband Subscription'], lw=4,
             label='宽带每一百人订阅比例增长率')
    plt.legend(loc='upper left')
    plt.show()

    plt.scatter(chinese_users['Year'], chinese_users['No. of Internet Users'])
    poly_reg = PolynomialFeatures(degree=3)
    x = chinese_users[['Year']]
    x_m = poly_reg.fit_transform(x)

    model_2 = linear_model.LinearRegression()
    model_2.fit(x_m, chinese_users[['No. of Internet Users']])
    plt.plot(x, model_2.predict(x_m),color='g')
    plt.show()

    pred_x = pd.DataFrame(np.arange(1980, 2051), columns=['Year'])
    pred_x_m = poly_reg.fit_transform(pred_x)
    plt.plot(pred_x, model_2.predict(pred_x_m))
    plt.show()


if __name__ == '__main__':
    # 准备环境
    prepare_profile()
    # 读取文件，获取全球互联网用户信息
    global_users = pd.read_csv('data/Final.csv', delimiter=',', usecols=range(1, 8))  # 由于第一列的列名未知，所以不使用第一列
    # 对全球用户进行分析：
    global_internet_users_analysis()
    entities_2020_internet_users_percentage_pie()
    entities_2020_internet_users_percentage_distribution_histogram()
    entities_2020_internet_users_percentage_distribution_scatter()
    draw_internet_users_percentage_annual_top_3_wordcloud()

    # 通过切片获取中国互联网用户信息
    chinese_users = global_users.loc[global_users['Entity'] == 'China']
    chinese_users_analysis()
