import shutil 
import os

# makes new ("clean") test  test_panels and test_results folder available for accepting a new image. 

shutil.rmtree("/home/ldalton/test_panels", ignore_errors=True, onerror=None)
os.makedirs("/home/ldalton/test_panels")

shutil.rmtree("/home/ldalton/test_results", ignore_errors=True, onerror=None)
os.makedirs("/home/ldalton/test_results")


shutil.rmtree("/home/ldalton/test", ignore_errors=True, onerror=None)
os.makedirs("/home/ldalton/test")
