from selenium import webdriver
import time
import random
#from webdriver_manager.chrome import ChromeDriverManager


driver = webdriver.Chrome('/Users/radwan/Downloads/chromedriver')

videos = [
'https://www.youtube.com/watch?v=diHR3tdK-3g&t=10s'
'https://www.youtube.com/watch?v=Z0Kef5Z37sQ'
'https://www.youtube.com/watch?v=M_iauns2AUc'
'https://www.youtube.com/watch?v=T5t69f4TQu8'

]


for i in range(20):
	print("Running the Video for {} time".format(i))
	random_video = random.randit(0,8)
	driver.get(videos[random_video])
	sleep_time = random.rantin(10, 20)
	time.sleep(sleep_time)

driver.quit()