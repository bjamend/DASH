from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from time import sleep
import matplotlib.pyplot as plt

def data_collection(element_1, element_2):

	print("\nSubmitting query to database...")

	# run in headless mode so no browser window pops up
	options = Options()
	options.headless = True

	# set up web driver
	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

	# navigate to SAGA database site
	driver.get("http://sagadatabase.jp/Retrieval/db.cgi")

	print("\nQuery submitted. Waiting on database...")

	# check upper limits for x
	xlimits = driver.find_element("name", "xupper")
	xlimits_drp = Select(xlimits)
	xlimits_drp.select_by_visible_text("Exclude")

	# check upper limits for y
	ylimits = driver.find_element("name", "yupper")
	ylimits_drp = Select(ylimits)
	ylimits_drp.select_by_visible_text("Exclude")

	# set what will be extracted as function of [Fe/H]
	yaxis_box = driver.find_element("name", "AX:yaxis_text")
	yaxis_box.send_keys(f"[{element_2.capitalize()}/{element_1.capitalize()}]", Keys.ENTER)

	# open up first row of data
	driver.find_element("xpath", "//*[text()='Link 1']").click();

	print("\nData obtained. Plotting results...")

	# switch over to new tab
	driver.switch_to.window(driver.window_handles[1])

	# extract data from page
	file = driver.find_element("xpath", "/html/body").text.split("\n")

	driver.quit()

	return file