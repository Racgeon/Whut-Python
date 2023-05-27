import pandas as pd
from matplotlib import pyplot as plt

# plt.rcParams['font.sans-serif'] = ['STXIHEI']
data=pd.read_excel(io='data/计算机智能学院2022级学生分流结果.xls')
# plt.hist(data['绩点'],rwidth=0.8)
# plt.show()
# data.sort_values(by='绩点', ascending=False,inplace=True)
# over_or_eq_4:pd.DataFrame=data.loc[data['绩点'] >= 4.0]
# school_groups=over_or_eq_4.groupby('学院名称')
# count_df=pd.DataFrame()
# for school,school_df in school_groups:
#     count=school_df.count(axis=0)['序号']
#     new_df=pd.DataFrame({'人数':{school_df['学院名称'].iloc[0]:count}})
#     count_df=pd.concat([new_df,count_df],join='outer',axis=0)
# print(count_df)
# plt.rcParams['figure.figsize'] = (16, 8)
# plt.pie(count_df['人数'],labels=count_df.index,labeldistance=1.5)
# plt.show()

#       序号            学号   姓名 性别     学院名称        原班级  拟录取专业     绩点  拟分配班级
# 10    11  122207781320  汪佳惠  女     汽车学院    车辆类2213    计算机  4.270    NaN
# 17    18  122212740108  霍佳蕾  男     航运学院     海事2201    计算机  4.280    NaN
# 24    25  122210880519  赵子龙  男  计算机智能学院  计算机类m2205    计算机  4.220    NaN
# 25    26  122210880311   陈章  男  计算机智能学院  计算机类m2203    计算机  4.205    NaN
# 151  152  122207780307  高健洋  男     汽车学院    车辆类2203  计算机zy  4.280    NaN
# 153  154  122212380128  俞嘉怡  女     航运学院     导航2201  计算机zy  4.280    NaN
# 254  255  122203950209  李子豪  男   安全应急学院    管科类2202     软件  4.250    NaN
# 255  256  122203950212  李朱睿  男   安全应急学院    管科类2202     软件  4.290    NaN
# 256  257  122203950219  张哲瑞  男   安全应急学院    管科类2202     软件  4.240    NaN
# 272  273  122210880402  陈肇伟  男  计算机智能学院  计算机类m2204     软件  4.283    NaN
# 273  274  122210880324  李永辉  男  计算机智能学院  计算机类m2203     软件  4.243    NaN
# 274  275  122210880502  朱子贤  男  计算机智能学院  计算机类m2205     软件  4.242    NaN
# 325  326  122203950113   胡磊  男   安全应急学院    管科类2201   软件zy  4.270    NaN
# 400  401  122204951619  曾福文  男     机电学院    机电类2216    大数据  4.220    NaN
# 408  409  122210880134  肖连雲  男  计算机智能学院  计算机类m2201    大数据  4.267    NaN
# 409  410  122210880413   汪洋  女  计算机智能学院  计算机类m2204    大数据  4.267    NaN
# 480  481  122210880110  代文博  男  计算机智能学院  计算机类m2201   人工智能  4.294    NaN
print(data.loc[(data['绩点'] > 4.2) & (data['绩点'] < 4.3)& (data['学院名称']=='计算机智能学院')])