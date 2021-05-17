import warnings
warnings.filterwarnings("ignore")
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import statsmodels.api as sm


def main(datafile):
    #read dataset
    df=pd.read_csv(datafile,)
    #drop index column
    df.drop(df.columns[0],inplace=True, axis=1)

    df.info()

    df['Clothing ID']=df['Clothing ID'].astype('category')

    df['Clothing ID'].describe()

    #looking at the percentage of missing values in data frame
    df.isnull().sum()
    df.isnull().sum()/df.shape[0]*100

    df['Class Name'].unique()


    df['Clothing ID'].describe()

    #since Review test will be the base for sentiment analysis, only  deal with missing value for Review Text, Division Name,Class Name.

    #remove missing value for these columns since there are no other information to help to fill the missing value and the percentages are small.
    for column in ["Review Text","Division Name","Department Name","Class Name"]:
        df=df[df[column].notnull()]

    df.isnull().sum()/df.shape[0]

    # Create New Variables:
    #Word length
    df["Word Count"]= df['Review Text'].str.split().apply(len)

    #Character length
    df["Character Count"] = df['Review Text'].apply(len)

    df.info()

    fig, ax = plt.subplots(1,3, figsize=(12,4), sharey= False)
    sns.distplot(df.Age, ax=ax[0])
    ax[0].set_title("Age Distribution")
    ax[0].set_ylabel("Density")
    sns.distplot(df["Word Count"], ax=ax[1])
    ax[1].set_title("Word Count per Review Distribution")
    ax[1].set_ylabel("Density")
    sns.distplot(df["Character Count"], ax=ax[2])
    ax[2].set_title("Character Coun per Review Distribution")
    ax[2].set_ylabel("Density")

    #Age Distribution by the Usual Suspects. Round them up
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)
    x_var = "Age"
    plot_df = df['Age']
    for i, y in enumerate(["Rating", "Department Name", "Recommended IND"]):
        for x in set(df[y][df[y].notnull()]):
            sns.kdeplot(plot_df[df[y] == x], label=x, shade=False, ax=axes[i])
        axes[i].set_xlabel("{}".format(x_var))
        axes[i].set_title('{} Distribution by {}'.format(x_var, y))
        axes[i].legend()
    axes[0].set_ylabel('Occurrence Density')
    plt.show()


    for x in set(df["Class Name"][df["Class Name"].notnull()]):
        ax=sns.kdeplot(plot_df[df["Class Name"]==x], label=x, shade=False)

    ax.set(xlabel='Age', ylabel='Occurrence Density')
    ax.set_title('{} Distribution by {}'.format(x_var, "Class Name"))
    ax.legend(bbox_to_anchor=(1.01, 1),
               borderaxespad=0)
    plt.show()

    #percentage standardize
    def percentstandardize_barplot(x,y,hue, data, ax=None, order= None):
        """
        Standardize by percentage the data using pandas functions, then plot using Seaborn.
        Function arguments are and extention of Seaborns'.
        """
        sns.barplot(x= x, y=y, hue=hue, ax=ax, order=order,
        data=(data[[x, hue]]
         .reset_index(drop=True)
         .groupby([x])[hue]
         .value_counts(normalize=True)
         .rename('Percentage').mul(100)
         .reset_index()
         .sort_values(hue)))
        plt.title("Percentage Frequency of {} by {}".format(hue,x))
        plt.ylabel("Percentage %")

    huevar = "Recommended IND"
    f, axes = plt.subplots(1,2,figsize=(12,5))
    percentstandardize_barplot(x="Department Name",y="Percentage", hue=huevar,data=df, ax=axes[0])
    axes[0].set_title("Percentage Frequency of {}\nby Department Name".format(huevar))
    axes[0].set_ylabel("Percentage %")
    percentstandardize_barplot(x="Division Name",y="Percentage", hue=huevar,data=df, ax=axes[1])
    axes[1].set_title("Percentage Frequency of {}\nby Division Name".format(huevar))
    axes[1].set_ylabel("")
    plt.show()

    #Rating by Department and Divison Name
    xvar = ["Department Name","Division Name"]
    huevar = "Rating"
    f, axes = plt.subplots(1,2,figsize=(12,5))
    percentstandardize_barplot(x=xvar[0],y="Percentage", hue=huevar,data=df, ax=axes[0])
    axes[0].set_title("Percentage Frequency of {}\nby {}".format(huevar, xvar[0]))
    axes[0].set_ylabel("Percentage %")
    percentstandardize_barplot(x=xvar[1],y="Percentage", hue="Rating",data=df, ax=axes[1])
    axes[1].set_title("Percentage Frequency of {}\nby {}".format(huevar, xvar[1]))
    plt.show()


    #Correlating Average Rating and Recommended IND by Clothing ID
    temp = (df.groupby('Clothing ID')[["Rating","Recommended IND","Positive Feedback Count" ,"Age"]]
            .aggregate(['count','mean']))
    temp.columns = ["Rating Count","Rating Mean","Recommended IND Count",
                    "Recommended Mean","Positive Feedback Count","Positive Feedback Mean","Age Count","Age Mean"]
    temp.drop(["Recommended IND Count","Age Count"], axis=1, inplace =True)

    # Plot Correlation Matrix
    f, ax = plt.subplots(figsize=[9,6])
    ax = sns.heatmap(temp.corr()
        , annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title("Correlation Matrix for Mean and Count for\nRating,Recommended, and Age\nGrouped by Clothing ID")
    plt.show()


    g = sns.jointplot(x= "Recommended Mean",y='Rating Mean',data=temp,
                      kind='reg', color='b')
    plt.subplots_adjust(top=0.999)
    g.fig.suptitle("Rating Mean and Recommended Mean\nGrouped by Clothing ID")
    plt.show()

    g = sns.jointplot(x= "Positive Feedback Mean",y='Rating Mean',data=temp,
                      kind='reg', color='b')
    plt.subplots_adjust(top=0.999)
    g.fig.suptitle("Rating Mean and Recommended Mean\nGrouped by Clothing ID")
    plt.show()


    key = "Class Name"
    temp = (df.groupby(key)[["Rating","Recommended IND", "Age"]]
            .aggregate(['count','mean']))
    temp.columns = ["Count","Rating Mean","Recommended Likelihood Count",
                    "Recommended Likelihood","Age Count","Age Mean"]
    temp.drop(["Recommended Likelihood Count","Age Count"], axis=1, inplace =True)

    # Plot Correlation Matrix
    f, ax = plt.subplots(figsize=[9,6])
    ax = sns.heatmap(temp.corr()
        , annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title("Correlation Coefficient for Mean and Count for\nRating, Recommended Likelihood, and Age\nGrouped by {}".format(key))
    plt.show()
    print("Class Categories:\n",df["Class Name"].unique())



    # Simple Linear Regression Model
    model_fit = sm.OLS(temp["Recommended Likelihood"],
                   sm.add_constant(temp["Age Mean"])).fit()
    temp['resid'] = model_fit.resid

    # Plot
    g = sns.jointplot(y="Recommended Likelihood",x='Age Mean',data=temp,
                      kind='reg', color='r')
    plt.subplots_adjust(top=0.999)
    g.fig.suptitle("Age Mean and Recommended Likelihood\nGrouped by Clothing Class")
    plt.ylim(.7, 1.01)

    # Annotate Outliers
    head = temp.sort_values(by=['resid'], ascending=[False]).head(2)
    tail = temp.sort_values(by=['resid'], ascending=[False]).tail(2)

    def ann(row):
        ind = row[0]
        r = row[1]
        plt.gca().annotate(ind, xy=( r["Age Mean"], r["Recommended Likelihood"]),
                xytext=(2,2) , textcoords ="offset points", )

    for row in head.iterrows():
        ann(row)
    for row in tail.iterrows():
        ann(row)

    plt.show()
    del head, tail

    temp[temp["Recommended Likelihood"] > .95]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Exploratory')
    parser.add_argument('--path', type=str, default="Womens Clothing E-Commerce Reviews.csv",
                        help='path to womens clothing dataset')
    args = parser.parse_args()

    main(args.path)
