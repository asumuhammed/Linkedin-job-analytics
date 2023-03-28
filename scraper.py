import pandas as pd
import numpy as np
import json
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

#login linkdin function which will take username and password  
def login(username, password):
    url = "https://www.linkedin.com/jobs/search?trk=guest_homepage-basic_guest_nav_menu_jobs&position=1&pageNum=0"
    #opening linkdin on the crome browser
    driver.get(url)
    #finding and click on the signing button
    driver.find_element(By.XPATH,"/html/body/div[1]/header/nav/div/a[2]").click()
    #finding username id and password id 
    username_id = driver.find_element(By.ID, "username")
    password_id = driver.find_element(By.ID, "password")
    # pass the credentials to username and password
    username_id.send_keys(username)
    password_id.send_keys(password)
    #click on the login button
    driver.find_element(By.XPATH,'//*[@id="organic-div"]/form/div[3]/button').click()
    driver.implicitly_wait(20)
    #finding job button and click on it
    print(f'{username} succesfully logedin')
    return 

# defining a function which collect all the job links
def page_link():
    # creating a blank list which will going to store all the job links
    Link = []
    # searching the list of all jobs 
    jobs_block = driver.find_element(By.CLASS_NAME,'jobs-search-results-list')
    # search all the jobs belongs in the jobs list
    jobs_list= driver.find_elements(By.CSS_SELECTOR, '.jobs-search-results__list-item')
    # creating a loop which will iterate through all the jobs and find the link and store in the Link
    for job in jobs_list:
        try:
            # As we know link of any company will store in the a(anchor tag) so finding anchor tag here
            links = job.find_element(By.TAG_NAME,"a")
            # collecting the link from the anchor tag and appending link into Link 
            Link.append(links.get_attribute("href"))
            # this will scroll down our page one by one elements
            driver.execute_script("arguments[0].scrollIntoView();", links)   ## scroll down by each step 
            # sleep time
            time.sleep(3)
        except:
            pass
    return Link

# defining a function which will collect links from different pages it will take page_number arguments
def get_links(page_number):
    all_links = []
    # Running a loop which will move us from first page to last page one by one
    try:
        for i in range(1,page_number+1):
            # click on the page number getting from the loop
            driver.find_element(By.XPATH,f"//button[@aria-label='Page {i}']").click() ## click on page 1,2,3..
            # giving the time to out driver so it can work smoothly
            driver.implicitly_wait(10)
            all_links += page_link()
            print(f'{i} page links extracted')
    except:
        pass
    return all_links

# defining a function which will take a link as arg and find scraper all the job details
def job_info(link):
    # opening the link
    driver.get(link)
    # provide time to driver to be update properly
    time.sleep(3)
    driver.implicitly_wait(10)
    # finding the butoon for see more 
    try:
        button = driver.find_element(By.XPATH,"/html/body/div[5]/div[3]/div/div[1]/div[1]/div/div[2]/footer/button")
    except:
        button = driver.find_element(By.XPATH,"/html/body/div[5]/div[3]/div/div[1]/div[1]/div/div[4]/footer/button")
    #click in the see more
    driver.execute_script("arguments[0].click();", button)
    # try to scrape city ,state, country from 
    try:
        location= driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__bullet').text
    except:
        location = ''
    # try to scrape job title 
    try:
        job_title = driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__job-title').text
    except:
        job_title = ''
    # try to scrape company name
    try:
        company_name = driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__company-name').text
    except:
        company_name = ''
    # try to scrape industry
    try:
        industry = driver.find_element(By.XPATH,'/html/body/div[5]/div[3]/div/div[1]/div[1]/div/section/section/div[1]/div[2]').text
    except:
        industry = ''
    #tyr to scrape employee count
    try:
        employee_count = driver.find_element(By.CLASS_NAME,'jobs-company__inline-information').text
    except:
        employee_count = ''
    # try to scrape followers
    try:
        followers = driver.find_element(By.CLASS_NAME,'artdeco-entity-lockup__subtitle').text
    except:
        followers = ''
    # try to scrape involvement
    try:
        involvement = driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__job-insight').text
    except:
        involvement = ''
    # try to scrape applicants1
    try:
        applicant1 = driver.find_elements(By.CLASS_NAME, "jobs-unified-top-card__bullet")[1].text
    except:
        applicant1 = ''
        # try to scrape applicants2
    try:
        applicant2 = driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__job-insight--highlight').text
    except:
        applicant2 = ''
        # try to scrape job description
    try:
        job_description = driver.find_element(By.CLASS_NAME,'jobs-description__container').text
    except:
        job_description = ''
    try:
        workplace_type = driver.find_element(By.CLASS_NAME,'jobs-unified-top-card__workplace-type').text
    except:
        workplace_type = ''
    # returning all the information in the form of dictonary
    return {'company_name': company_name, 'job_title' : job_title, 'involvement': involvement, 'workplace_type': workplace_type,
           'industry': industry, 'location': location,  'employee_count': employee_count, 'followers_count': followers,
           'applicants_1': applicant1, 'applicants_2': applicant2, 'job_description': job_description, 'link': link}
    
# defining a function which will take range as input and using upper function return of list of dict of job details
def get_job_info(start, end):
# creating a function which will take range and return list of dictonary containig all the job information
    data = []
    for i in range(start, end):
        try:
            # append the job info in the form of dictonary to data
            data.append(job_info(df['Link'][i]))
            print(f'file-{i} done')
        except:
            pass
    print("! Task Completed !")
    return data 


#open crome browser
driver = webdriver.Chrome(executable_path=r"C:\api\chromedriver.exe")

#calling the login function to login the linkdin account
username = input("enter username : ")
password = input("enter password : ")

login(username,password)
driver.maximize_window()

#calling the get_links function and collect all the links
all_links = get_links(40)
# converting list of links into dataframe
df = pd.DataFrame({'Link':all_links})
# exporting dataframe into jsonfile 
df.to_csv('links.csv')

#calling function get_job_info and storing all the job details in the dataframe called all_data
all_data = pd.DataFrame(get_job_info(0, len(df)))
all_data.to_csv('full_data.csv')
