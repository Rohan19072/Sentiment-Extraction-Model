from flask import Flask,render_template,request
import pickle,os,time
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.colors as mcolors
import matplotlib as mpl
from flask_mysqldb import MySQL
from preprocess import pre_process
from sklearn.pipeline import Pipeline
import mysql.connector as connection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


os.chdir("d:\Rohan\SEM8\Flask")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def function(text):
    model = pickle.load(open('model.pkl','rb'))
    sentiment = str(model.predict([text])).replace('[','').replace(']','').replace("'",'').replace("'",'')
    return sentiment

def bargraph(df,imagepath):
    # Calculate value counts of sentiment labels
    values = df['sentiment'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize the values for colormap
    norm = mcolors.Normalize(vmin=0,vmax=max(values))

    colours = ['#64C2A6', '#AADEA7', '#E6F69D']

     # Colormap - Build the colour maps
    cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)


    # Create bar chart with normalized values and colormap
    bar1 = ax.bar(values.index, values, width=0.6, color=cmap(norm(values)))

    # Add labels to the bars with custom font size and color
    ax.bar_label(bar1, labels=[f'{e}' for e in values], padding=3, color='black', fontsize=14)

    # Remove spines
    ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)

    # Add a horizontal line above the chart with custom color and linewidth
    ax.plot([0.12, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='#E3120B', linewidth=0)
    

    # Add a title with custom font size, weight, and alignment
    ax.text(x=0.12, y=.93, s="Sentiment Value Counts of the Reviews", transform=fig.transFigure, ha='left', fontsize=20, weight='bold', alpha=.8)

    # Set background color of the figure
    fig.patch.set_facecolor('white')

    # Adjust layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.savefig(imagepath)

def piechart(df2,imagepath2):
    plt.figure(figsize=(8,8))
    plt.title("Sentiment Analysis of Reviews",pad=10)
    
    image = df2.sentiment.value_counts().plot.pie(shadow=True,autopct='%1.2f%%',textprops={'fontsize':14},colors = ['#ffdfba', '#a2c8e2', '#d2e9ca'])
    plt.legend()
    plt.savefig(imagepath2)

def get_5rating(driver):
    try:
        five_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[1]/td[3]/span[2]/a').text
    except:
        try:
            five_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[1]/td[3]/span[2]').text
        except:
            five_star_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[4]/div/div/table/tbody/tr[1]/td[3]/a').text
            
    return five_star_rating

def get_4rating(driver):
    try:
        four_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[2]/td[3]/span[2]/a').text
    except:
        try:
            four_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[2]/td[3]/span[2]').text
        except:
            four_star_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[4]/div/div/table/tbody/tr[2]/td[3]/a').text

    return four_star_rating

def get_3rating(driver):
    try:
        three_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[3]/td[3]/span[2]/a').text
    except:
        try:
            three_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[3]/td[3]/span[2]').text
        except:
            three_star_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[4]/div/div/table/tbody/tr[3]/td[3]/a').text
    
    return three_star_rating

def get_2rating(driver):
    try:
        two_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[4]/td[3]/span[2]/a').text
    except:
        try:
            two_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[4]/td[3]/span[2]').text
        except:
            two_star_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[4]/div/div/table/tbody/tr[4]/td[3]/a').text

    return two_star_rating

def get_1rating(driver):
    try:
        one_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[5]/td[3]/span[2]/a').text
    except:
        try:
            one_star_rating = driver.find_element(By.XPATH,'//*[@id="histogramTable"]/tbody/tr[5]/td[3]/span[2]').text
        except:
            one_star_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[4]/div/div/table/tbody/tr[5]/td[3]/a').text

    return one_star_rating

def get_overall_rating(driver):
    try:
        overall_rating = driver.find_element(By.XPATH,'//*[@id="reviewsMedley"]/div/div[1]/span[1]/span/div[2]/div/div[2]/div/span/span').text
    except:
        overall_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[2]/div/div[2]/div/span/span').text    
    return overall_rating

def get_global_rating(driver):
    try:
        global_rating = driver.find_element(By.XPATH,'//*[@id="reviewsMedley"]/div/div[1]/span[1]/span/div[3]/span').text
    except:
        global_rating = driver.find_element(By.XPATH,'//*[@id="cm_cr_dp_d_rating_histogram"]/div[3]/span').text    
    return global_rating

@app.route('/',methods =['POST'])
def predict():
    input_value = [request.form.get('Text')]
    model = pickle.load(open('model.pkl','rb'))
    prediction = str(model.predict(input_value)).replace('[','').replace(']','').replace("'",'').replace("'",'')
    return render_template('demo.html',Category = " {} Sentiment ".format(prediction))

@app.route('/submit',methods = ['POST'])
def submit():
    if request.method == "POST":
        file = request.files['file']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath = os.path.join('static',file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath,encoding='ISO-8859-1',header=None)
        df.rename(columns = {0:'text'},inplace=True)

        data = df.astype(str)

        data['sentiment'] = data['text'].apply(function)

        imagepath = os.path.join('static','image'+'.png')
        bargraph(data,imagepath)
    return render_template('file.html',image=imagepath)

@app.route('/submit2',methods = ['POST'])
def submit2():
    if request.method == "POST":
        url = request.form.get('url')

        if url:
        
            driver = webdriver.Chrome()
            driver.get(url)
            wait_time = 10

            df2 = pd.DataFrame(columns=['Title','Body'])

            REVIEW_BTN_Xpath = '//*[@id="reviews-medley-footer"]/div[2]/a'
            REVIEW_BTN = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, REVIEW_BTN_Xpath)))

            overall_rating = get_overall_rating(driver)
            total_global_rating = get_global_rating(driver)
            five_star_rating = get_5rating(driver)
            four_star_rating = get_4rating(driver)
            three_star_rating = get_3rating(driver)
            two_star_rating = get_2rating(driver)
            one_star_rating = get_1rating(driver)

            REVIEW_BTN.click()
            
            while True:
                try:
                    html = driver.page_source
                    soup = BeautifulSoup(html,'html.parser')
                    reviews = soup.find_all('div',attrs={'data-hook':'review'})

                    for review in reviews:
                        title = review.find('a',attrs={'data-hook':'review-title'}).text.strip()
                        body = review.find('span',attrs={'data-hook':'review-body'}).text.strip()
                        df2 = df2.append({'Title':title,'Body':body},ignore_index=True)

                    next_page = driver.find_element(By.XPATH,'//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a').click()
                    time.sleep(2)

                except:
                    break  
            
            print("Fetched Reviews Successfully")
            
            driver.quit()

            df2['text'] = df2['Title'] + ' ' + df2['Body']

            df2['sentiment'] = df2['text'].apply(function)

            df2.drop(['Title','Body'],axis=1,inplace=True)
            df2.to_csv('reviews.csv')
            total = len(df2.index)

            data={'overall':overall_rating,'global':total_global_rating,'total':total,'five_stars':five_star_rating,'four_stars':four_star_rating,'three_stars':three_star_rating,'two_stars':two_star_rating,'one_stars':one_star_rating}

            mydb = connection.connect(host='localhost',database='sql demo',user='root',passwd='',use_pure=True)
            cursor = mydb.cursor()
            
            for i,row in df2.iterrows():
                sql = "INSERT INTO training_data (`text`,`sentiment`) VALUES(%s,%s)"
                cursor.execute(sql, tuple(row))
                mydb.commit()

            mydb.close()

            imagepath2 = os.path.join('static','reviews_analysis'+'.jpg')
            piechart(df2,imagepath2)
        

        else:
            return "INVALID URL"
    
    return render_template('reviews_analysis.html',image=imagepath2,data=data)

@app.route('/click',methods=['GET'])
def click():
    if request.method == 'GET':
        mydb = connection.connect(host='localhost',database='sql demo',user='root',passwd='',use_pure=True)
        query = "SELECT * FROM training_data;"
        data = pd.read_sql(query,mydb)

        # data = pd.read_csv('train.csv',encoding='ISO-8859-1')
        data.dropna(subset=['text'], axis=0, inplace=True)

        pipe_lr = Pipeline(steps=[('tf',TfidfVectorizer(preprocessor=pre_process)),('lr',RandomForestClassifier())])

        X = data['text']
        Y = data['sentiment']
        
        pipe_lr.fit(X,Y)

        pickle.dump(pipe_lr,open("model.pkl","wb"))
        
        mydb.close()

    return "Model Trained successfully"  

if __name__ == "__main__":
    app.run(debug=True,port=3004,use_reloader=False )
    




