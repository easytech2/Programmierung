from selenium import webdriver
from time import sleep

driver = webdriver.Chrome(excutetable_path="/User/Radwan/Downloads/chromedriver.exe")
driver.get("https:/instagram.com")
sleep(4)
driver.find_element_by_xpath("//input[@name=\"username\"]")\
	.send_keys("put your username here")
driver.find_element_by_xpath("//input[@name=\"password\"]")\
	.send_keys("put your password here")
driver.find_element_by_xpath("//button[@type=\"submit\"]")\
	.cklick()

sleep(3)



